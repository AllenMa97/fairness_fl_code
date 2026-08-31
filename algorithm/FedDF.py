# FedDF: Ensemble Distillation for Robust Model Fusion in Federated Learning
# https://arxiv.org/abs/2011.09063 (NeurIPS 2020)
# 核心思想: 操作空间为输出分布空间（ensemble logit averaging + KD）。
# 需要一批无标签代理数据（随机噪声或公共数据）。各客户端模型在代理数据上产生 logits，
# 服务器将其平均后作为 teacher，蒸馏训练全局模型。
# Core Idea: Operates in output distribution space (ensemble logit averaging + KD).
# Requires unlabeled proxy data (random noise or public data). Client models produce logits
# on proxy data; the server averages them as teacher to distill the global model.

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
from tool.tensorboard_logger import log_test_metrics, log_system_metrics, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor


def _generate_proxy_data(param_dict, device, num_samples=128):
    """根据任务类型生成代理数据（随机噪声 / 公共数据占位）。"""
    if "SENT_CLF" in param_dict["task"]:
        max_len = param_dict.get('max_len', 128)
        input_ids = torch.randint(0, 30522, (num_samples, max_len), device=device)
        attention_mask = torch.ones((num_samples, max_len), device=device)
        return {'type': 'SENT', 'input_ids': input_ids, 'attention_mask': attention_mask}
    elif "IMG_CLF" in param_dict["task"]:
        inp_ch = 3 if param_dict.get('dataset', '').lower() not in ['fmnist', 'mnist'] else 1
        imgs = torch.randn(num_samples, inp_ch, 32, 32, device=device)
        return {'type': 'IMG', 'imgs': imgs}
    else:  # Tabular_CLF
        inp_size = param_dict.get('nn_input_size', 128)
        X = torch.randn(num_samples, inp_size, device=device)
        return {'type': 'TAB', 'X': X}


def _model_logits(model, param_dict, proxy, device):
    """给定模型和代理数据，返回 logits（未经 softmax/sigmoid）。"""
    if proxy['type'] == 'SENT':
        input_ids = proxy['input_ids']
        attention_mask = proxy['attention_mask']
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        return logits
    elif proxy['type'] == 'IMG':
        preds, _ = model(proxy['imgs'])
        # IMG/Tabular 使用 BCE，preds 为 sigmoid 后概率 → 用 logit 反变换
        return torch.logit(torch.clamp(preds, min=1e-7, max=1 - 1e-7))
    else:  # TAB
        if "ANN" in str(type(model)):
            preds, _ = model(proxy['X'])
        elif "LogisticRegression" in str(type(model)):
            preds = model(proxy['X'])
        else:
            preds = model(proxy['X'])
        return torch.logit(torch.clamp(preds, min=1e-7, max=1 - 1e-7))


def _train_single_client_feddf(client_id, device, model, param_dict,
                                training_dataloaders, algorithm_epoch_T,
                                accumulation_steps, use_amp, scaler, criterion,
                                basic_path, iter_t, communication_round_I, num_clients_K):
    """FedDF 单客户端训练：标准本地训练。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    gpu_seconds = 0

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0
        epoch_total_size = 0

        for batch_id, batch in enumerate(client_i_dataloader):
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_loss = criterion(logits, labels)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
                labels = batch["labels"].to(device)
                preds, _ = model(imgs)
                batch_loss = criterion(preds[:, 0], labels.float())
            else:
                X = batch["X"].to(device)
                labels = batch["labels"].to(device)
                if "ANN" in str(type(model)):
                    preds, _ = model(X)
                else:
                    preds = model(X)
                batch_loss = criterion(preds[:, 0], labels.float())

            true_batch_size = labels.size(0)
            epoch_total_size += true_batch_size
            gpu_start = time.time()
            with autocast_context(device, use_amp):
                loss = torch.sum(batch_loss) / true_batch_size
            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)
                model.zero_grad()
            gpu_seconds += (time.time() - gpu_start)
            epoch_total_loss += loss
            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}; Epoch {epoch+1}; Loss: {epoch_total_loss/max(epoch_total_size,1):.4f}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    return {'gpu_seconds': gpu_seconds}


def Fed_DF(device,
           global_model,
           algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
           training_dataloaders,
           training_dataset,
           client_dataset_list,
           param_dict,
           testing_dataloader,
           testing_dataset_len,
           start_round=0):
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()

    # FedDF 超参数
    df_T = float(param_dict.get('FedDF_T', 3.0))           # KD 温度
    df_alpha = float(param_dict.get('FedDF_alpha', 0.5))   # KD 损失权重
    df_steps = int(param_dict.get('FedDF_steps', 20))      # 服务器蒸馏步数
    df_proxy = int(param_dict.get('FedDF_proxy', 128))     # 代理样本数

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t+1}; Select clients: {idxs_users}; Start FedDF Local Training")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_feddf,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ===== 聚合模型 (FedAvg 权重) =====
        logger.info("FedDF: Aggregation + Ensemble KD on proxy data")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        client_models = []
        for i, cid in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            sm = torch.load(client_model_path, weights_only=False)
            cp = get_parameters(sm)
            theta_list.append(cp)
            client_models.append(sm)
            updates = {}
            for j, (p_l, p_g) in enumerate(zip(cp, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_l) - torch.tensor(p_g)
            client_model_updates.append(updates)

        weights = [client_datasets_size_list[j] for j in idxs_users]
        theta_arr = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_arr, axis=0, weights=weights).tolist()
        set_parameters(global_model, theta_avg)

        # ===== FedDF 核心：Ensemble Logit Distillation =====
        gpu_s = time.time()
        global_model.to(device)
        for m in client_models:
            m.to(device).eval()
        global_model.train()

        proxy = _generate_proxy_data(param_dict, device, df_proxy)

        # 1) 计算各客户端 teacher logits → ensemble average
        with torch.no_grad():
            logits_list = []
            for m in client_models:
                lg = _model_logits(m, param_dict, proxy, device)
                logits_list.append(lg.unsqueeze(0))
            teacher_logits = torch.cat(logits_list, dim=0).mean(dim=0)  # 集成平均

        # 2) 蒸馏训练全局模型若干步
        distill_optim = BERTCLF_Optimizer(method="ADAM", learning_rate=param_dict['learning_rate']*0.1, max_grad_norm=0)
        distill_optim.set_parameters(list(global_model.named_parameters()))

        for _s in range(df_steps):
            student_logits = _model_logits(global_model, param_dict, proxy, device)

            if "SENT_CLF" in param_dict["task"]:
                # KD: soft label CE + hard label CE（此处 proxy 无标签，仅 soft KD）
                student_soft = torch.log_softmax(student_logits / df_T, dim=1)
                teacher_soft = torch.softmax(teacher_logits / df_T, dim=1).detach()
                loss_kd = -torch.sum(teacher_soft * student_soft, dim=1).mean() * (df_T ** 2)
            else:
                # BCE 场景：KL / MSE between sigmoid(logits/T)
                s_p = torch.sigmoid(student_logits / df_T)
                t_p = torch.sigmoid(teacher_logits / df_T).detach()
                loss_kd = torch.nn.functional.mse_loss(s_p, t_p) * (df_T ** 2)

            distill_optim.zero_grad()
            (df_alpha * loss_kd).backward()
            distill_optim.step()

        # 清理临时模型
        for m in client_models:
            m.to("cpu")
            del m
        client_models.clear()
        global_model.to("cpu")

        total_gpu_seconds += (time.time() - gpu_s)
        del theta_arr, theta_list
        gc.collect()

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(f"Round {iter_t+1} Test; GPU total: {total_gpu_seconds:.1f}s, Avg: {avg_gpu_seconds:.1f}s")

        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "IMG_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO; HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            else:  # Tabular
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO; HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()

        log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t+1, client_model_updates=client_model_updates)

        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(param_dict=param_dict, iter_t=iter_t, global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time)
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("FedDF training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedDF.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
