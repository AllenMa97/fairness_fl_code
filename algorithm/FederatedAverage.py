# FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data
# https://arxiv.org/abs/1602.05629
# 核心思想：基于数据量加权的模型聚合，是联邦学习的基础算法

import os
import gc
import time
import torch
from collections import OrderedDict
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.client_parallel import ClientParallelExecutor
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config


def _needs_client_updates(monitoring_config, step):
    """Return whether this communication step needs per-client update tensors."""
    frequency = max(1, int(monitoring_config.get('gradient_freq', 1)))
    return bool(monitoring_config.get('gradient')) and step % frequency == 0


def _aggregate_state_dicts(client_states, weights):
    """Compute a dataset-weighted FedAvg state on CPU without NumPy copies."""
    if not client_states or len(client_states) != len(weights):
        raise ValueError("client_states and weights must be non-empty and have equal length")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("aggregation weights must sum to a positive value")

    keys = list(client_states[0].keys())
    if any(list(state.keys()) != keys for state in client_states[1:]):
        raise ValueError("all client state dictionaries must have identical keys")

    averaged = OrderedDict()
    for name in keys:
        reference = client_states[0][name].detach().cpu()
        if reference.is_floating_point() or reference.is_complex():
            value = torch.zeros_like(reference, device='cpu')
            for state, weight in zip(client_states, weights):
                value.add_(state[name].detach().to(device='cpu', dtype=reference.dtype),
                           alpha=float(weight) / total_weight)
            averaged[name] = value
        elif reference.dtype == torch.bool:
            averaged[name] = reference.clone()
        else:
            # Match the old NumPy path: average integer buffers, then cast on load.
            value = torch.zeros_like(reference, device='cpu', dtype=torch.float64)
            for state, weight in zip(client_states, weights):
                value.add_(state[name].detach().to(device='cpu', dtype=torch.float64),
                           alpha=float(weight) / total_weight)
            averaged[name] = value.to(dtype=reference.dtype)
    return averaged


def _build_client_updates(client_states, reference_state):
    """Build CPU weight deltas in the format expected by gradient monitoring."""
    keys = list(reference_state.keys())
    if any(list(state.keys()) != keys for state in client_states):
        raise ValueError("client and reference state dictionaries must have identical keys")

    updates = []
    for state in client_states:
        client_update = OrderedDict()
        for index, name in enumerate(keys):
            local_value = state[name].detach().cpu()
            reference_value = reference_state[name].detach().to(
                device='cpu', dtype=local_value.dtype)
            if local_value.dtype == torch.bool:
                delta = local_value.to(torch.int8) - reference_value.to(torch.int8)
            else:
                delta = local_value - reference_value
            client_update[str(index)] = delta
        updates.append(client_update)
    return updates


def _train_single_client_fedavg(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 iter_t, communication_round_I, num_clients_K):
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

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

    # Move the trained state to CPU so GPU residency remains bounded by the executor.
    client_state = model.cpu().state_dict()
    return {'gpu_seconds': gpu_seconds, 'state_dict': client_state}


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

        cfg_deep = get_monitoring_config(param_dict)
        step = iter_t + 1
        client_states = [result.pop('state_dict') for result in results]
        aggregation_weights = [client_datasets_size_list[client_id] for client_id in idxs_users]

        # Client deltas are only materialized on rounds that actually log them.
        client_model_updates = None
        if _needs_client_updates(cfg_deep, step):
            reference_state = OrderedDict(
                (name, value.detach().cpu().clone())
                for name, value in global_model.state_dict().items()
            )
            client_model_updates = _build_client_updates(client_states, reference_state)
            del reference_state

        logger.info("Parameter aggregation")
        averaged_state = _aggregate_state_dicts(client_states, aggregation_weights)
        logger.info("Update Global Model")
        global_model.load_state_dict(averaged_state, strict=True)
        del averaged_state, client_states
        gc.collect()

        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        # 非最后一轮做测试（外层会做最后测试并记录到final/）
        if (iter_t + 1) != param_dict['communication_round_I']:
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

        # ===== 深度监控（每轮都执行，包括最后一轮）=====
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