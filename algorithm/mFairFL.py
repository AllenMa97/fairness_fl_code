# mFairFL: Multi-dimensional Fair Federated Learning
# https://arxiv.org/abs/2312.05551
# 核心思想：通过多目标优化框架同时优化模型的准确性和多个公平性指标

import copy
import os
import gc
import random
import time
import torch
import math
import numpy as np
import traceback

from tool.logger import *
from tool.aggregation import average_state_dicts
from tool.utils import get_parameters, set_parameters, FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from hypothesis.generator import LatentGenerator, FigGenerator
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step, AutocastSafeBCELoss
from tool.client_parallel import ClientParallelExecutor
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.experiment_state import (
    AlgorithmRunResult,
    capture_training_loader_generator_states,
    restore_training_loader_generator_states,
)


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def _train_single_client_mfairfl(client_id, device, model, param_dict, training_dataloaders,
                                  algorithm_epoch_T, client_datasets_size_list, num_clients_K,
                                  basic_path, iter_t, communication_round_I,
                                  accumulation_steps, use_amp, scaler, criterion,
                                  local_Lagrangian_list, lambda_param_optimizer_list):
    """mFairFL 单客户端训练函数（供 ClientParallelExecutor 调用）"""
    client_i_aggregation_weight = 0  # 由调用方计算

    logger.info(f"Client {client_id} Init Local Model By Copy From Global Model")
    model.train()
    model.to(device)

    lambda_param = local_Lagrangian_list[client_id]
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))

    lambda_param_optimizer = lambda_param_optimizer_list[client_id]

    client_i_dataloader = training_dataloaders[client_id]

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0
        epoch_total_size = 0

        for batch_id, batch in enumerate(client_i_dataloader):
            labels = batch["labels"].to(device)
            protecteds = batch["protected"]
            true_batch_size = labels.size()[0]
            epoch_total_size += true_batch_size
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)

            labels = batch["labels"].to(device)
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

                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
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

            group_flag = protecteds.gt(0.5)
            one_batch_group_1_count = sum(group_flag)
            one_batch_group_0_count = true_batch_size - sum(group_flag)
            if (one_batch_group_1_count != 0) and (one_batch_group_0_count != 0):
                one_batch_group_1_avg_loss = sum(batch_loss[group_flag]) / one_batch_group_1_count
                one_batch_group_0_avg_loss = (sum(batch_loss) - sum(batch_loss[group_flag])) / one_batch_group_0_count
                one_batch_group_avg_loss_gap = torch.abs(one_batch_group_0_avg_loss - one_batch_group_1_avg_loss)
                if float(batch_id) % 50 == 0:
                    logger.info(f"Origin task loss：{loss.item()} ;\n"
                                f"one_batch_group_avg_loss_gap: {one_batch_group_avg_loss_gap.item()} ;\n"
                                f"in batch_id:{batch_id} of epoch:{epoch} in Client:{client_id}.")
                loss += lambda_param * one_batch_group_avg_loss_gap

            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)

                if (one_batch_group_1_count != 0) and (one_batch_group_0_count != 0):
                    grad_lambda = torch.autograd.grad(
                        outputs=lambda_param * one_batch_group_avg_loss_gap,
                        inputs=lambda_param,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True
                    )[0]
                    lambda_param.grad = -grad_lambda
                    lambda_param_optimizer.step()

                    local_Lagrangian_list[client_id] = lambda_param

                model.zero_grad()

            gpu_end_time = time.time()

            epoch_total_loss += float(loss.detach())

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, labels, batch_loss, loss
            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                del imgs, labels, batch_loss, loss

            gc.collect()
            torch.cuda.empty_cache()

        # 收尾：最后一批不满一个累积窗口时先 step 残余梯度，避免永不更新
        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

        torch.cuda.empty_cache()
        gc.collect()

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {'gpu_seconds': gpu_end_time - gpu_start_time}


# AAAI2024 mFairFL
def mFairFL(device,
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
    accumulation_steps = int(256 / param_dict['batch_size'])
    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]


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

    local_Lagrangian_list = [torch.nn.Parameter(torch.tensor(1.), requires_grad=True) for _ in range(num_clients_K)] # 预设多个拉格朗日乘子
    lambda_param_optimizer_list = [torch.optim.SGD([local_Lagrangian_list[i]], lr=0.1) for i in range(num_clients_K)]


    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)

    total_gpu_seconds = 0.0 if resume_state is None else float(resume_state.total_gpu_seconds)
    previous_state = {} if resume_state is None else dict(resume_state.algorithm_state)
    if resume_state is not None:
        restore_training_loader_generator_states(
            training_dataloaders, previous_state.get("training_loader_generator_states")
        )
    users_gpu_seconds_list = list(previous_state.get("users_gpu_seconds_list", [0.0] * num_clients_K))
    if len(users_gpu_seconds_list) != num_clients_K:
        raise RuntimeError("mFairFL checkpoint has incompatible users_gpu_seconds_list")
    total_communication_cost = 0.0 if resume_state is None else float(resume_state.total_communication_cost)
    client_selection_history = [] if resume_state is None else [list(row) for row in resume_state.client_selection_history]
    runtime_offset = 0.0 if resume_state is None else float(resume_state.total_runtime_seconds)
    runtime_start = time.monotonic()

    # 断点恢复：拉格朗日乘子必须就地(in-place)写回以保持 optimizer 绑定
    if resume_state is not None:
        saved_lagrangian = previous_state.get("local_Lagrangian_list")
        if saved_lagrangian is not None:
            if len(saved_lagrangian) != len(local_Lagrangian_list):
                raise RuntimeError("mFairFL checkpoint has incompatible local_Lagrangian_list")
            for lagrangian_param, value in zip(local_Lagrangian_list, saved_lagrangian):
                lagrangian_param.data.copy_(
                    torch.tensor(value, dtype=lagrangian_param.dtype, device=lagrangian_param.device)
                )

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parallel_executor = ClientParallelExecutor(device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
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
            # style="FedProx",
        )

        selected_clients = idxs_users.tolist() if hasattr(idxs_users, "tolist") else list(idxs_users)
        client_selection_history.append([int(client_id) for client_id in selected_clients])

        selected_client_training_dataset_size = sum([client_datasets_size_list[item] for item in idxs_users])
        average_weight = [0 for _ in range(num_clients_K)]
        for id in idxs_users:
            average_weight[id] = client_datasets_size_list[id] / selected_client_training_dataset_size
        average_weight = np.array(average_weight)

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")


        # Simulate Client Parallel
        client_results = parallel_executor.run_clients(
            idxs_users, _train_single_client_mfairfl,
            param_dict=param_dict,
            training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T,
            client_datasets_size_list=client_datasets_size_list,
            num_clients_K=num_clients_K,
            basic_path=basic_path,
            iter_t=iter_t,
            communication_round_I=communication_round_I,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion,
            local_Lagrangian_list=local_Lagrangian_list,
            lambda_param_optimizer_list=lambda_param_optimizer_list,
        )
        for i, result in enumerate(client_results):
            id = idxs_users[i]
            users_gpu_seconds_list[id] += result['gpu_seconds']

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        total_communication_cost += len(idxs_users) * 2 * model_MB_size

        # Global operation
        # 更新delta
        logger.info(f"Communication Round {(iter_t + 1)} "
                    f"Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []

        # 先读取正常客户的参数
        client_states = []
        aggregation_weights = []

        for id in idxs_users:

            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            client_params = get_parameters(selected_model)
            client_states.append(selected_model.state_dict())
            aggregation_weights.append(client_datasets_size_list[id]) # 这个地方只需要读取客户的数据量，不用除以总量！

            # 计算该客户端的更新量
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            del selected_model
            gc.collect()
        try:
            if (len(aggregation_weights) != 0) and (sum(aggregation_weights) != 0):
                logger.info("Parameter aggregation")
                aggregated_state = average_state_dicts(client_states, weights=aggregation_weights)

                logger.info("Update Global Model with aggregated parameters")
                global_model.load_state_dict(aggregated_state, strict=True)
                del client_states, aggregated_state
                gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

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
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
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
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
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
                    communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
                )
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                                   communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                                   selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
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
                    "local_Lagrangian_list": [float(param.item()) for param in local_Lagrangian_list],
                },
                amp_scaler=scaler,
                total_gpu_seconds=total_gpu_seconds,
                total_runtime_seconds=runtime_offset + (time.monotonic() - runtime_start),
                total_communication_cost=total_communication_cost,
                client_selection_history=client_selection_history,
                start_time=start_time,
            )

            # 清理旧检查点，保留最近 N 个
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_mFairFL.pt")
    torch.save(global_model, save_path)
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state={
            "users_gpu_seconds_list": users_gpu_seconds_list,
            "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
            "local_Lagrangian_list": [float(param.item()) for param in local_Lagrangian_list],
        },
        amp_scaler_state=None if scaler is None else scaler.state_dict(),
        client_selection_history=client_selection_history,
    )
