import os
import gc
import time
import copy
import torch
import numpy as np
from tool.logger import *
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor


def _train_single_client_fedprox(client_id, device, model, param_dict,
                                  training_dataloaders, algorithm_epoch_T,
                                  use_amp, scaler, criterion,
                                  basic_path, iter_t, communication_round_I, num_clients_K,
                                  global_model_ref, miu):
    """FedProx 单客户端训练函数（可被 ClientParallelExecutor 并行调度）"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    gpu_seconds = 0

    # Local Training
    for epoch in range(algorithm_epoch_T):
        # 设置状态变量
        epoch_total_loss = 0
        epoch_total_size = 0

        for batch in client_i_dataloader:
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)

            labels = batch["labels"].to(device)

            # FedProx独有: 直接用模型参数计算proximal term，保留梯度
            proximal_term = 0.
            for (name_g, param_g), (name_l, param_l) in zip(global_model_ref.named_parameters(), model.named_parameters()):
                if param_l.requires_grad:
                    try:
                        proximal_term += torch.norm(param_g.to(device) - param_l, p=2) ** 2
                    except Exception:
                        pass

            true_batch_size = labels.size()[0]
            epoch_total_size += true_batch_size

            gpu_start_time = time.time()

            with autocast_context(device, use_amp):
                if "SENT_CLF" in param_dict["task"]:
                    features, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    activated_preds = logits
                    _, preds = torch.max(activated_preds, dim=1)
                    batch_loss = criterion(activated_preds, labels)

                elif "IMG_CLF" in param_dict["task"]:
                    preds, features = model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    if "ANN" in str(type(model)):
                        local_prediction, features = model(X)
                    elif "LogisticRegression" in str(type(model)):
                        local_prediction = model(X)
                    else:
                        local_prediction = model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                loss = torch.sum(batch_loss) / true_batch_size
                loss += ((miu / 2) * proximal_term) / true_batch_size

            scale_backward(loss, scaler)
            scaler_step(scaler, optimizer)

            gpu_end_time = time.time()
            gpu_seconds += (gpu_end_time - gpu_start_time)

            model.zero_grad()
            epoch_total_loss += loss

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, labels
            elif "IMG_CLF" in param_dict["task"]:
                del imgs, labels

            gc.collect()

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    return {'gpu_seconds': gpu_seconds}


def Fed_Prox(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0
            ):
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

    try:
        miu = param_dict["miu"]
    except Exception:
        miu = 1
    logger.info(f"Now the μ of FedProx is: {miu}")

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 初始化客户端并行执行器
    parallel_executor = ClientParallelExecutor(
        device=device,
        global_model=global_model,
        param_dict=param_dict,
        needs_global_model_during_training=False,
    )

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
            style="FedProx",
        )

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        # 客户端训练（自动串行/并行）
        results = parallel_executor.run_clients(
            idxs_users,
            _train_single_client_fedprox,
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
            global_model_ref=global_model,
            miu=miu,
        )

        # 收集 GPU 计时
        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []

        # Global operation
        logger.info("Parameter aggregation")
        theta_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            client_params = get_parameters(selected_model)
            theta_list.append(client_params)

            # 计算该客户端的更新量
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        # FedAvg新版论文的聚合权重是数据占比
        # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
        # 如一个weights = [w1, w2, w3, w4]
        # 那么结果就是(theta1 * w1 + theta2 * w2 + theta3 * w3 + theta4 * w4)/ sum(w1+w2+w3+w4)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        # FedAvg旧版论文的聚合权重是平均
        # theta_avg = np.mean(theta_list, 0).tolist()
        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)

        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                                   testing_dataset_len)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                   selected_client_count=len(idxs_users), model_mb_size=model_MB_size)
                flush()

                # ===== 深度监控 =====
                cfg_deep = get_monitoring_config(param_dict)
                if (iter_t + 1) % max(1, cfg_deep.get('deep_log_freq', 5)) == 0:
                    log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t + 1, client_model_updates=client_model_updates)
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                   selected_client_count=len(idxs_users), model_mb_size=model_MB_size)
                flush()

                # ===== 深度监控 =====
                cfg_deep = get_monitoring_config(param_dict)
                if (iter_t + 1) % max(1, cfg_deep.get('deep_log_freq', 5)) == 0:
                    log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t + 1, client_model_updates=client_model_updates)
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                   selected_client_count=len(idxs_users), model_mb_size=model_MB_size)
                flush()

                # ===== 深度监控 =====
                cfg_deep = get_monitoring_config(param_dict)
                if (iter_t + 1) % max(1, cfg_deep.get('deep_log_freq', 5)) == 0:
                    log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t + 1, client_model_updates=client_model_updates)

            # 保存检查点（按 checkpoint_save_freq 间隔）
            if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
                save_checkpoint(
                    param_dict=param_dict,
                    iter_t=iter_t,
                    global_model=global_model,
                    total_gpu_seconds=total_gpu_seconds,
                    client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                    start_time=start_time
                )

                # 清理旧检查点，保留最近 N 个
                clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedProx.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost