import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.client_parallel import ClientParallelExecutor
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config


def _train_single_client_fedavg(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K):
    """FedAvg 单客户端训练函数（可被 ClientParallelExecutor 并行调度）"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    gpu_seconds = 0

    # Local Training
    for epoch in range(algorithm_epoch_T):
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
                    batch_loss = criterion(preds[:,0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    if "ANN" in str(type(model)):
                        local_prediction, features = model(X)
                    elif "LogisticRegression" in str(type(model)):
                        local_prediction = model(X)
                    else:
                        local_prediction = model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

            loss = torch.sum(batch_loss) / true_batch_size
            scale_backward(loss, scaler)

            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)
                model.zero_grad()

            gpu_end_time = time.time()
            gpu_seconds += (gpu_end_time - gpu_start_time)

            epoch_total_loss += loss

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, labels
            elif "IMG_CLF" in param_dict["task"]:
                del imgs, labels

            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

    # 保存训练后的模型
    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    return {'gpu_seconds': gpu_seconds}


def Fed_AVG(device,
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
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))

    # AMP 初始化：根据 param_dict['use_amp'] 决定是否启用混合精度
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    gc.collect()

    basic_path = param_dict['model_path']

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 初始化客户端并行执行器
    parallel_executor = ClientParallelExecutor(
        device=device,
        global_model=global_model,
        param_dict=param_dict,
        needs_global_model_during_training=False,
    )

    for iter_t in range(start_round, communication_round_I):
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

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        # 客户端训练（自动串行/并行）
        results = parallel_executor.run_clients(
            idxs_users,
            _train_single_client_fedavg,
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

        # 收集 GPU 计时
        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # ── 收集客户端模型更新（用于梯度监控中的客户端方差/余弦相似度）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []  # [{name: tensor_delta}, ...]
        
        # Global operation
        logger.info("Parameter aggregation")
        theta_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_params = get_parameters(selected_model)
            theta_list.append(client_params)
            
            # 计算该客户端的更新量（近似梯度方向）
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            
            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()

        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)

        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

        # 每轮都做测试和监控
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
            logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
            
            # ===== TensorBoard logging =====
            log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size)
            log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(),
                model_mb_size=model_MB_size)
            flush()
            
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
            FR = 1-DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                        f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            
            # ===== TensorBoard logging =====
            log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                FR=float(FR), HM=float(HM),
                step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size)
            log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(),
                model_mb_size=model_MB_size)
            flush()
            
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            
            # ===== TensorBoard logging =====
            log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                FR=float(FR), HM=float(HM),
                step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size)
            log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(),
                model_mb_size=model_MB_size)
            flush()

        # ===== 深度监控（每轮都执行）=====
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
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedAvg.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost