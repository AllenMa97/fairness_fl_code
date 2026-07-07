# SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
# https://arxiv.org/abs/1910.06378
# 核心思想：使用控制变量（Control Variates）来校正客户端漂移，提高联邦学习的收敛速度和稳定性
import gc
import os
import time
import copy
import torch
from tool.logger import *
from algorithm.Optimizers import Scaffold_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import (FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF,
                        get_HM_by_two_value, get_parameters)
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor


def _train_single_client_scaffold(client_id, device, model, param_dict,
                                   training_dataloaders, algorithm_epoch_T,
                                   accumulation_steps, use_amp, scaler, criterion,
                                   basic_path, iter_t, communication_round_I, num_clients_K):
    """Extracted single-client training logic for Scaffold with parallel support."""
    gpu_seconds = 0.0

    # 提前保存一份没经过训练的全局模型的参数
    x = copy.deepcopy(model)

    model.train()
    model.to(device)
    optimizer = Scaffold_Optimizer(model.parameters(), method=param_dict['optimize_method'],
                                     learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    # Local Training
    for epoch in range(algorithm_epoch_T):
        # 设置状态变量
        epoch_total_loss = 0
        epoch_total_size = 0

        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)

            labels = batch["labels"].to(device)

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
                epoch_total_loss += float(loss)
            scale_backward(loss, scaler)

            if (batch_id + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step(device, x.control, model.control)
                model.zero_grad()

            gpu_end_time = time.time()
            gpu_seconds += (gpu_end_time - gpu_start_time)

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, labels
            elif "IMG_CLF" in param_dict["task"]:
                del imgs, labels
            gc.collect()
            torch.cuda.empty_cache()

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")
        torch.cuda.empty_cache()

    # 更新参数ci
    temp = {}
    for k, v in model.named_parameters():
        temp[k] = v.data

    gpu_start_time = time.time()
    for k, v in x.named_parameters():
        model.control[k] = model.control[k].to(device)
        x.control[k] = x.control[k].to(device)
        v.data = v.data.to(device)
        temp[k] = temp[k].to(device)

        model.control[k] = model.control[k] - x.control[k] + (v.data - temp[k]) / (
                algorithm_epoch_T * 0.005)
        model.delta_y[k] = temp[k] - v.data
        model.delta_control[k] = model.control[k] - x.control[k]

        model.control[k] = model.control[k].cpu()
        x.control[k] = x.control[k].cpu()
        v.data = v.data.cpu()
        temp[k] = temp[k].cpu()
    gpu_end_time = time.time()
    gpu_seconds += (gpu_end_time - gpu_start_time)

    # Upgrade the local model list
    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    del x, temp, model
    gc.collect()
    torch.cuda.empty_cache()

    return {'gpu_seconds': gpu_seconds}


def Scaffold(device,
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
    accumulation_steps = int(256 / param_dict['batch_size'])

    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    gc.collect()

    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(num_clients_K) + "Clients")

    # Parameter Initialization
    # Scaffold论文提出的全局模型Learning Rate，论文中取1
    param_dict["slr"] = 1

    # 将Scaffold论文提出的所有控制变量都初始化为0
    for k, v in global_model.named_parameters():
        global_model.control[k] = torch.zeros_like(v.data).to(device)
        global_model.delta_control[k] = torch.zeros_like(v.data).to(device)
        global_model.delta_y[k] = torch.zeros_like(v.data).to(device)

    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)
    elif "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parallel_executor = ClientParallelExecutor(
        device=device,
        global_model=global_model,
        param_dict=param_dict,
        needs_global_model_during_training=True,  # Scaffold has shared control variables
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
            style="FedAvg",
        )

        logger.info(f"*** Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training! ***")

        # Simulate Client Parallel
        results = parallel_executor.run_clients(
            idxs_users,
            _train_single_client_scaffold,
            param_dict=param_dict,
            training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            scaler=scaler,
            criterion=criterion,
            basic_path=basic_path,
            iter_t=iter_t,
            communication_round_I=communication_round_I,
            num_clients_K=num_clients_K,
        )

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
        # Scaffold Initialization
        x = {}
        c = {}

        # Scaffold Aggregation
        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        for k, v in global_model.named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)
        # 记录GPU计算结束时间
        gpu_end_time = time.time()
        total_gpu_seconds += (gpu_end_time - gpu_start_time)
        for j in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(j + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化

            # 计算该客户端的更新量
            client_params = get_parameters(selected_model)
            updates = {}
            for idx_p, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(idx_p)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            # 记录GPU计算开始时间
            gpu_start_time = time.time()
            for k, v in selected_model.named_parameters():
                selected_model.delta_y[k] = selected_model.delta_y[k].to(device)
                selected_model.delta_control[k] = selected_model.delta_control[k].to(device)
                x[k] = x[k].to(device)
                c[k] = c[k].to(device)

                x[k] += selected_model.delta_y[k] / len(idxs_users)
                c[k] += selected_model.delta_control[k] / len(idxs_users)

                selected_model.delta_y[k] = selected_model.delta_y[k].cpu()
                selected_model.delta_control[k] = selected_model.delta_control[k].cpu()
                x[k] = x[k].cpu()
                c[k] = c[k].cpu()
            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            total_gpu_seconds += (gpu_end_time - gpu_start_time)
            torch.save(selected_model.cpu(), client_model_path)  # 持久化
            del selected_model
            gc.collect()
            torch.cuda.empty_cache()

        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        logger.info("Update Global Model")
        global_model.to(device)
        for k, v in global_model.named_parameters():
            x[k] = x[k].to(device)
            c[k] = c[k].to(device)
            # v.data += x[k].data * param_dict["slr"]
            v.data += x[k].data * 1
            global_model.control[k].data = global_model.control[k].data.to(device)
            global_model.control[k].data += c[k].data
            x[k] = x[k].cpu()
            c[k] = c[k].cpu()
            global_model.control[k].data = global_model.control[k].data.cpu()
            torch.cuda.empty_cache()

        global_model.cpu()
        torch.cuda.empty_cache()
        # 记录GPU计算结束时间
        gpu_end_time = time.time()
        total_gpu_seconds += (gpu_end_time - gpu_start_time)

        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del x, c
        gc.collect()
        torch.cuda.empty_cache()

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024))
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)),
                                   selected_client_count=len(idxs_users), model_mb_size=(sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)))
                flush()

            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1-DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                # ===== TensorBoard logging =====
                log_test_metrics(
                    accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024))
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)),
                                   selected_client_count=len(idxs_users), model_mb_size=(sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)))
                flush()

            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
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
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024))
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * (sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)),
                                   selected_client_count=len(idxs_users), model_mb_size=(sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)))
                flush()


        # ===== 深度监控（每轮都执行，包括最后一轮）=====
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, 
                         iter_t + 1, client_model_updates=client_model_updates)

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
    save_path = os.path.join(save_dir, f"global_Scaffold.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost