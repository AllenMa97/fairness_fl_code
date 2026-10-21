# FedProto: Federated Prototype Learning with Global-Local Knowledge Distillation
# https://arxiv.org/abs/2105.00243
# 核心思想：利用原型学习和全局-局部知识蒸馏来提高联邦学习的泛化能力

import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import get_parameters
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step, AutocastSafeBCELoss
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor
from tool.aggregation import average_state_dicts
from tool.experiment_state import (
    AlgorithmRunResult,
    capture_training_loader_generator_states,
    restore_training_loader_generator_states,
)


def Fed_PROTO(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0,
            resume_state=None
            ):
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))

    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)


    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    gc.collect()

    # basic_path = os.path.join("./save_path", param_dict['dataset_name'],
    #                           param_dict['split_strategy'],
    #                           param_dict['algorithm'],
    #                           param_dict['hypothesis'],
    #                           str(num_clients_K) + "Clients")
    basic_path = param_dict['model_path']


    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)

    # 断点恢复：恢复计数器、AMP、客户端选择历史与 DataLoader 生成器状态
    total_gpu_seconds = 0.0 if resume_state is None else float(resume_state.total_gpu_seconds)
    previous_state = {} if resume_state is None else dict(resume_state.algorithm_state)
    if resume_state is not None:
        restore_training_loader_generator_states(
            training_dataloaders, previous_state.get("training_loader_generator_states")
        )
    users_gpu_seconds_list = list(previous_state.get("users_gpu_seconds_list", [0.0] * num_clients_K))
    if len(users_gpu_seconds_list) != num_clients_K:
        raise RuntimeError("FedProto checkpoint has incompatible users_gpu_seconds_list")
    total_communication_cost = 0.0 if resume_state is None else float(resume_state.total_communication_cost)
    client_selection_history = [] if resume_state is None else [list(row) for row in resume_state.client_selection_history]
    runtime_offset = 0.0 if resume_state is None else float(resume_state.total_runtime_seconds)
    runtime_start = time.monotonic()

    num_of_class = 2
    # prototype_MB_size = sys.getsizeof(torch.rand([num_of_class, 768])) / (1024 ** 2)
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # 全局原型列表跨轮保留：本轮下发给客户端的是上一轮聚合出的全局原型，FedProto 的
    # 全局-局部知识蒸馏项依赖它。原实现在每轮开头清空列表，导致客户端训练期间
    # len(global_label_*_prototype_list) == 0 恒成立、蒸馏项恒为 0，算法退化成 FedAvg。
    # Global prototypes persist across rounds: clients receive the prototypes aggregated in
    # the previous round. Clearing the lists at the start of every round made the
    # distillation term identically zero, degenerating FedProto into FedAvg.
    global_label_0_prototype_list = list(previous_state.get("global_label_0_prototype_list", []))
    global_label_1_prototype_list = list(previous_state.get("global_label_1_prototype_list", []))

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for iter_t in range(start_round, communication_round_I):
        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        selected_client_training_dataset_size = sum([client_datasets_size_list[item] for item in idxs_users])
        selected_clients = idxs_users.tolist() if hasattr(idxs_users, "tolist") else list(idxs_users)
        client_selection_history.append([int(client_id) for client_id in selected_clients])
        average_weight = [0 for _ in range(num_clients_K)]
        for id in idxs_users:
            average_weight[id] = client_datasets_size_list[id] / selected_client_training_dataset_size
        average_weight = np.array(average_weight)

        # 每轮的原型素材累加器（仅本轮使用）；跨轮状态见 global_label_*_prototype_list
        global_label_0_feature_list = []
        global_label_1_feature_list = []

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        def _train_single_client_fedproto(client_id, device, model, param_dict, training_dataloaders, algorithm_epoch_T, 
                                        use_amp, scaler, criterion, basic_path, iter_t, communication_round_I, 
                                        num_clients_K, accumulation_steps, average_weight, client_datasets_size_list,
                                        global_label_0_prototype_list, global_label_1_prototype_list):
            client_i_aggregation_weight = average_weight[client_id]

            # Local Initialization
            # 下发模型
            logger.info(f"Client {client_id}: Copy From Global Model")
            model = copy.deepcopy(model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[client_id]

            client_i_label_0_feature_list = []
            client_i_label_1_feature_list = []

            gpu_seconds_for_client = 0

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                # for batch in client_i_dataloader:
                for batch_id, batch in enumerate(client_i_dataloader):
                    label_0_feature_list = []
                    label_1_feature_list = []

                    if "SENT_CLF" in param_dict["task"]:
                        # input_ids尺寸 [batch_size, max_len]
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                        imgs = batch["img"].to(device)
                    elif "Tabular_CLF" in param_dict["task"]:
                        X = batch["X"].to(device)

                    # labels尺寸 [batch_size]
                    labels = batch["labels"].to(device)

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

                    # 记录GPU计算开始时间
                    gpu_start_time = time.time()

                    with autocast_context(device, use_amp):
                        if "SENT_CLF" in param_dict["task"]:
                            # features尺寸 [batch_size, emb_dim]
                            # logits尺寸 [batch_size, category]
                            features, logits = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            # activated_preds = logits.softmax(dim=1)
                            activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                            _, preds = torch.max(activated_preds, dim=1)
                            # batch_loss尺寸 [batch_size]
                            batch_loss = criterion(activated_preds, labels)

                        elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                            # preds尺寸 [batch_size, 1]
                            preds, features = model(imgs)
                            batch_loss = criterion(preds[:, 0], labels.float())

                        elif "Tabular_CLF" in param_dict["task"]:
                            # local_prediction尺寸 [batch_size, 1]
                            if "ANN" in str(type(model)):
                                local_prediction, features = model(X)
                            elif "LogisticRegression" in str(type(model)):
                                local_prediction = model(X)
                            else:
                                local_prediction = model(X)
                            batch_loss = criterion(local_prediction[:, 0], labels.float())

                        loss = torch.sum(batch_loss) / true_batch_size

                    if "SENT_CLF" in param_dict["task"]:
                        del activated_preds, logits
                    gc.collect()
                    torch.cuda.empty_cache()

                    # 添加原型素材
                    features = features.cpu()
                    sent_label_flag = labels.gt(0.5)
                    for idx, feat in enumerate(features):
                        if sent_label_flag[idx]:
                            label_1_feature_list.append(feat)
                            client_i_label_1_feature_list.append(feat)
                        else:
                            label_0_feature_list.append(feat)
                            client_i_label_0_feature_list.append(feat)

                    if "SENT_CLF" in param_dict["task"]:
                        del labels, sent_label_flag
                    gc.collect()
                    torch.cuda.empty_cache()


                    # 计算gap
                    with autocast_context(device, use_amp):
                        (label_0_feature_gap, label_1_feature_gap) = 0, 0
                        # 全局-本地类原型的差异（梯度只经本地原型回传，全局原型为固定目标）
                        if len(label_0_feature_list) != 0:
                            label_0_prototype = torch.stack(label_0_feature_list, dim=0).mean(dim=0)
                            if len(global_label_0_prototype_list) != 0:
                                # 全局原型是上一轮聚合出的固定目标（已 detach），仅本地原型
                                # 参与求导，符合 FedProto 的 stop-gradient 语义。
                                label_0_feature_gap = torch.norm((global_label_0_prototype_list[-1].to(device) - label_0_prototype.to(device)),
                                                                 p=2)
                        if len(label_1_feature_list) != 0:
                            label_1_prototype = torch.stack(label_1_feature_list, dim=0).mean(dim=0)
                            if len(global_label_1_prototype_list) != 0:
                                label_1_feature_gap = torch.norm((global_label_1_prototype_list[-1].to(device) - label_1_prototype.to(device)),
                                                                 p=2)
                        lamda_list = [1, 1]  # FedPro思路
                        gap_list = [label_0_feature_gap, label_1_feature_gap]
                        for index, lamda in enumerate(lamda_list):
                            loss += lamda * gap_list[index]

                    scale_backward(loss, scaler)

                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        scaler_step(scaler, optimizer)
                        # 清空梯度
                        model.zero_grad()


                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()

                    gpu_seconds_for_client += (gpu_end_time - gpu_start_time)

                    # 收尾：最后一批不满一个累积窗口时先 step 残余梯度，避免永不更新
                    if (batch_id + 1) % accumulation_steps != 0 and (batch_id + 1) == len(client_i_dataloader):
                        scaler_step(scaler, optimizer)
                    # 清空梯度
                    model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += float(loss.detach())
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[client_id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask
                    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                        del imgs, labels

                    gc.collect()
                    torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {client_id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # 客户端原型由外层循环在收集 results 时统一加权聚合。本函数与父进程处于同一
            # Python 进程（ClientParallelExecutor 用 CUDA Stream 而非多进程），若在此处
            # 追加 global_label_*_feature_list，外层循环会再追加一次，使原型权重放大 2 倍。
            # Client prototypes are aggregated exactly once, by the parent loop.
            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
            # local_model_list[client_id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            del model
            gc.collect()
            torch.cuda.empty_cache()
            
            return gpu_seconds_for_client, client_i_label_0_feature_list, client_i_label_1_feature_list, client_i_aggregation_weight


        # 使用并行执行器处理客户端训练
        parallel_executor = ClientParallelExecutor(device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)
        
        results = parallel_executor.run_clients(
            idxs_users,
            _train_single_client_fedproto,
            param_dict=param_dict,
            training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T,
            use_amp=use_amp,
            scaler=scaler,
            criterion=criterion,
            basic_path=basic_path,
            iter_t=iter_t,
            communication_round_I=communication_round_I,
            num_clients_K=num_clients_K,
            accumulation_steps=accumulation_steps,
            average_weight=average_weight,
            client_datasets_size_list=client_datasets_size_list,
            global_label_0_prototype_list=global_label_0_prototype_list,
            global_label_1_prototype_list=global_label_1_prototype_list
        )
        
        # 收集每个客户端的GPU时间及原型数据
        for i, client_id in enumerate(idxs_users):
            gpu_seconds, client_i_label_0_feature_list, client_i_label_1_feature_list, client_i_aggregation_weight = results[i]
            users_gpu_seconds_list[client_id] += gpu_seconds
            
            # 处理原型数据
            if len(client_i_label_0_feature_list) != 0:
                client_i_label_0_prototype = torch.stack(client_i_label_0_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_label_0_feature_list.append(client_i_aggregation_weight * client_i_label_0_prototype)

            if len(client_i_label_1_feature_list) != 0:
                client_i_label_1_prototype = torch.stack(client_i_label_1_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_label_1_feature_list.append(client_i_aggregation_weight * client_i_label_1_prototype)

        # Communicate
        logger.info(f"Communicate: {(iter_t + 1)}")
        total_gpu_seconds += sum(users_gpu_seconds_list)
        total_communication_cost += len(idxs_users) * 2 * prototype_MB_size
        logger.info(f"Total's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2  * prototype_MB_size} MB")

        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        client_states = []
        client_weights = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_params = get_parameters(selected_model)
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            client_states.append(selected_model.state_dict())
            client_weights.append(client_datasets_size_list[id])
            del selected_model
            gc.collect()

        # Global operation
        logger.info("Prototype aggregation")
        (global_label_0_prototype, global_label_1_prototype) = 0, 0

        # 前面已经乘过权重了，所以这里只需要加起来即可。
        # 全局原型以「脱离计算图 + CPU」的形式入列：一是 FedProto 把全局原型当作跨轮
        # 固定目标，不需要把梯度回传到上一轮的图；二是 checkpoints 会对 algorithm_state
        # 做 deepcopy，而带 grad_fn 的非叶张量不支持 deepcopy，会直接抛 RuntimeError。
        # Store the global prototypes detached and on CPU: FedProto treats them as fixed
        # targets across rounds without gradients, and checkpoints deepcopy algorithm_state,
        # which rejects non-leaf tensors that still carry a grad_fn.
        if len(global_label_0_feature_list) != 0:
            for proto in global_label_0_feature_list:
                global_label_0_prototype += proto
            global_label_0_prototype_list.append(global_label_0_prototype.detach().to("cpu").clone())  # 更新全局的各种原型
        if len(global_label_1_feature_list) != 0:
            for proto in global_label_1_feature_list:
                global_label_1_prototype += proto
            global_label_1_prototype_list.append(global_label_1_prototype.detach().to("cpu").clone())  # 更新全局的各种原型

        # ── 模型聚合：FedProto 与 FedAvg 一致地聚合全局骨干模型（含 BN buffer）──
        # 原实现遗漏此步，导致 global_model 自初始化后从未被更新（客户端每轮都从
        # 同一初始模型出发），评测必然塌缩。详见 tool/aggregation.py 头部说明。
        logger.info("Model aggregation")
        averaged_state = average_state_dicts(client_states, weights=client_weights)
        global_model.load_state_dict(averaged_state, strict=True)
        del averaged_state, client_states, client_weights
        gc.collect()


        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I'] and (iter_t + 1) % param_dict.get('test_eval_freq', 1) == 0:
            if "SENT_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                                   testing_dataset_len)
                accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * prototype_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * prototype_MB_size,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=prototype_MB_size)
                flush()

            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * prototype_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * prototype_MB_size,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=prototype_MB_size)
                flush()


            elif "Tabular_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * prototype_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * prototype_MB_size,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=prototype_MB_size)
                flush()


        # ===== 深度监控（每轮都执行，包括最后一轮）=====
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, 
                         iter_t + 1, client_model_updates=client_model_updates)

        # 保存检查点（按 checkpoint_save_freq 间隔，含全局原型）
        if (param_dict.get('checkpoint_save_freq', 1) > 0
                and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0
                and (iter_t + 1) < communication_round_I):
            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                algorithm_state={
                    "users_gpu_seconds_list": users_gpu_seconds_list,
                    "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
                    "global_label_0_prototype_list": global_label_0_prototype_list,
                    "global_label_1_prototype_list": global_label_1_prototype_list,
                },
                amp_scaler=scaler,
                total_gpu_seconds=total_gpu_seconds,
                total_runtime_seconds=runtime_offset + (time.monotonic() - runtime_start),
                total_communication_cost=total_communication_cost,
                client_selection_history=client_selection_history,
                start_time=start_time
            )

            # 清理旧检查点，保留最近 N 个
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedProto.pt")
    torch.save(global_model, save_path)
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state={
            "users_gpu_seconds_list": users_gpu_seconds_list,
            "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
            "global_label_0_prototype_list": global_label_0_prototype_list,
            "global_label_1_prototype_list": global_label_1_prototype_list,
        },
        amp_scaler_state=None if scaler is None else scaler.state_dict(),
        client_selection_history=client_selection_history,
    )
