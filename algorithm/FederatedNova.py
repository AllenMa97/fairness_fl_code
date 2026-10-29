# FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization
# https://arxiv.org/abs/2007.07481
# 核心思想：通过归一化异构更新来解决目标不一致问题，使用动量项来加速收敛
import gc
import os
import time
import math
import copy
import torch
import numpy as np
from tool.logger import *
from tool.aggregation import average_state_dicts
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step, AutocastSafeBCELoss
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor
from tool.experiment_state import (
    AlgorithmRunResult,
    capture_training_loader_generator_states,
    restore_training_loader_generator_states,
)


def Fed_Nova(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0,
            resume_state=None,
            ):
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    gc.collect()

    # 与其它算法一致：优先用 setup 阶段已建好（含 variant_tag 隔离）的 model_path
    basic_path = param_dict.get('model_path') or os.path.join(
        "./save_path", param_dict['dataset_name'],
        param_dict['split_strategy'],
        param_dict['algorithm'],
        param_dict['hypothesis'],
        str(num_clients_K) + "Clients")

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        client_dir = os.path.join(basic_path, "client_" + str(k + 1))
        os.makedirs(client_dir, exist_ok=True)
        full_path = os.path.join(client_dir, 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化
    # FedNova算法中，需要计算一共进行了多少个Batch的更新
    local_update_times_list = [int(math.floor(size / param_dict['batch_size'])) for size in
                               client_datasets_size_list]


    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)

    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)

    # 断点恢复：恢复计数器、AMP、客户端选择历史与 DataLoader 生成器状态
    total_gpu_seconds = 0.0 if resume_state is None else float(resume_state.total_gpu_seconds)
    previous_state = {} if resume_state is None else dict(resume_state.algorithm_state)
    if resume_state is not None:
        restore_training_loader_generator_states(
            training_dataloaders, previous_state.get("training_loader_generator_states")
        )
    users_gpu_seconds_list = list(previous_state.get("users_gpu_seconds_list", [0.0] * num_clients_K))
    if len(users_gpu_seconds_list) != num_clients_K:
        raise RuntimeError("FedNova checkpoint has incompatible users_gpu_seconds_list")
    total_communication_cost = 0.0 if resume_state is None else float(resume_state.total_communication_cost)
    client_selection_history = [] if resume_state is None else [list(row) for row in resume_state.client_selection_history]
    runtime_offset = 0.0 if resume_state is None else float(resume_state.total_runtime_seconds)
    runtime_start = time.monotonic()

    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(start_round, communication_round_I):
        # 先选客户端，只对选中的客戶下发模型
        # Client Selection
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        selected_client_training_dataset_size = sum([client_datasets_size_list[item] for item in idxs_users])
        average_weight = [0 for _ in range(num_clients_K)]
        for id in idxs_users:
            average_weight[id] = client_datasets_size_list[id] / selected_client_training_dataset_size
        average_weight = np.array(average_weight)

        # FedNova论文Eq.12的“括号”--(  ∑{_i=1}{^m} p_i * τ{_i}{t}  )
        sacling_fator = sum([average_weight[index] * local_update_times_list[index] for index in idxs_users])

        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        selected_clients = idxs_users.tolist() if hasattr(idxs_users, "tolist") else list(idxs_users)
        client_selection_history.append([int(client_id) for client_id in selected_clients])

        def _train_single_client_fednova(client_id, device, model, param_dict, training_dataloaders, algorithm_epoch_T, 
                                       use_amp, scaler, criterion, basic_path, iter_t, communication_round_I, 
                                       num_clients_K, sacling_fator, local_update_times_list, client_datasets_size_list):
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

            gpu_seconds_for_client = 0

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                # for batch_index, batch in enumerate(client_i_dataloader):
                for batch in client_i_dataloader:
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
                            # features尺寸 [batch_size, emb_dim]
                            preds, features = model(imgs)
                            batch_loss = criterion(preds[:,0], labels.float())

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

                        # 注意FedNova需要对参数变化量进行归一化操作，这里对损失函数进行改造（参考原文Eq.12）
                        normalized_loss = loss * sacling_fator / local_update_times_list[client_id]
                        # 防止出现INF和NAN
                        if not (torch.isinf(normalized_loss).any() or torch.isnan(normalized_loss).any()):
                            loss = normalized_loss

                    scale_backward(loss, scaler)

                    # FedAvg算法一个batch就做一次更新
                    scaler_step(scaler, optimizer)

                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()

                    gpu_seconds_for_client += (gpu_end_time - gpu_start_time)

                    # 清空梯度
                    model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += float(loss.detach())
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[client_id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels
                    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                        del imgs, labels
                    gc.collect()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {client_id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
            # local_model_list[client_id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            del model
            gc.collect()
            # torch.cuda.empty_cache()
            
            return gpu_seconds_for_client


        # 使用并行执行器处理客户端训练
        parallel_executor = ClientParallelExecutor(device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)
        
        results = parallel_executor.run_clients(
            idxs_users=idxs_users,
            train_fn=_train_single_client_fednova,
            # 注意：run_clients 内部已按 (client_id, device, model_copy) 注入前三个位置参数，
            # 这里不能再传 device/model，否则触发 "got multiple values for argument 'device'"
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
            sacling_fator=sacling_fator,
            local_update_times_list=local_update_times_list,
            client_datasets_size_list=client_datasets_size_list
        )
        
        # 收集每个客户端的GPU时间
        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        total_communication_cost += len(idxs_users) * 2 * model_MB_size
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")


        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []

        # Global operation
        logger.info("Parameter aggregation")
        client_states = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            client_params = get_parameters(selected_model)
            client_states.append(selected_model.state_dict())

            # 计算该客户端的更新量
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            del selected_model
            gc.collect()

        aggregated_state = average_state_dicts(client_states, weights=[client_datasets_size_list[j] for j in idxs_users])
        logger.info("Update Global Model")
        global_model.load_state_dict(aggregated_state, strict=True)
        del client_states, aggregated_state

        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        gc.collect()

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
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024))
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)),
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=(sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)))
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
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024))
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)),
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=(sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)))
                flush()


            elif "Tabular_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
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
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024))
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)),
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=(sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)))
                flush()


        # ===== 深度监控（每轮都执行，包括最后一轮）=====
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, 
                         iter_t + 1, client_model_updates=client_model_updates)

        # 保存检查点（按 checkpoint_save_freq 间隔）
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
    save_path = os.path.join(save_dir, f"global_FedNova.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state={
            "users_gpu_seconds_list": users_gpu_seconds_list,
            "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
        },
        amp_scaler_state=None if scaler is None else scaler.state_dict(),
        client_selection_history=client_selection_history,
    )
