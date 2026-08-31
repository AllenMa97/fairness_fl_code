# FedLGD: Federated Learning on Virtual Heterogeneous Data with Local-global Distillation
# https://arxiv.org/abs/2303.02278
# 核心思想: 操作空间为梯度空间（gradient matching）。在服务器端利用客户端上传的本地梯度，
# 通过在虚拟异构数据上进行本地-全局蒸馏来弥合客户端数据异质性差异。
# Core Idea: Operates in gradient space (gradient matching). The server matches gradients
# on virtual heterogeneous data via local-global distillation to mitigate data heterogeneity.

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


def _task_forward(model, param_dict, batch, device, criterion):
    """统一任务前向，返回 (loss_per_sample, features)，保持梯度。"""
    if "SENT_CLF" in param_dict["task"]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        features, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_loss = criterion(logits, labels)
        return batch_loss, features, labels, input_ids, attention_mask
    elif "IMG_CLF" in param_dict["task"]:
        imgs = batch["img"].to(device)
        labels = batch["labels"].to(device)
        preds, features = model(imgs)
        batch_loss = criterion(preds[:, 0], labels.float())
        return batch_loss, features, labels, imgs, None
    else:  # Tabular_CLF
        X = batch["X"].to(device)
        labels = batch["labels"].to(device)
        if "ANN" in str(type(model)):
            local_prediction, features = model(X)
        elif "LogisticRegression" in str(type(model)):
            local_prediction = model(X)
            features = X
        else:
            local_prediction, features = model(X) if hasattr(model, 'shared_base') else (model(X), X)
        batch_loss = criterion(local_prediction[:, 0], labels.float())
        return batch_loss, features, labels, X, None


def _train_single_client_fedlgd(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K):
    """FedLGD 单客户端训练函数，训练结束后返回本地梯度字典（近似）用于服务器梯度匹配。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    gpu_seconds = 0
    # 保存初始参数用于计算上传的梯度近似值 (w_local - w_global)
    initial_params = [torch.tensor(p.data.clone()) for p in model.parameters()]

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0
        epoch_total_size = 0

        for batch_id, batch in enumerate(client_i_dataloader):
            batch_loss, features, labels, inp1, inp2 = _task_forward(model, param_dict, batch, device, criterion)
            true_batch_size = labels.size(0)
            epoch_total_size += true_batch_size

            gpu_start_time = time.time()
            with autocast_context(device, use_amp):
                loss = torch.sum(batch_loss) / true_batch_size
            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)
                model.zero_grad()

            gpu_end_time = time.time()
            gpu_seconds += (gpu_end_time - gpu_start_time)
            epoch_total_loss += loss

            del features, labels
            gc.collect()

        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        avg_loss = epoch_total_loss / max(epoch_total_size, 1)
        logger.info(f"Round {iter_t + 1}/{communication_round_I}; Client {client_id}/{num_clients_K}; "
                    f"Epoch {epoch + 1}; Avg Loss: {avg_loss:.4f}")

    # 计算上传到服务器的近似本地梯度（delta = w_trained - w_initial）
    local_delta = {}
    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            local_delta[name] = (p.data.detach().cpu() - initial_params[i].cpu()).numpy()

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    return {'gpu_seconds': gpu_seconds, 'local_delta': local_delta}


def Fed_LGD(device,
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

    # FedLGD 超参数
    lgd_lambda = float(param_dict.get('FedLGD_lambda', 1.0))       # 梯度匹配损失权重
    lgd_steps = int(param_dict.get('FedLGD_steps', 5))             # 服务器端梯度匹配步数
    lgd_lr = float(param_dict.get('FedLGD_lr', 0.01))              # 服务器端学习率
    num_virtual = int(param_dict.get('FedLGD_virtual_samples', 8)) # 每类虚拟样本数

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (FedLGD)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedlgd,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型参数（FedAvg 风格加权平均）
        logger.info("FedLGD: Parameter aggregation + Gradient matching")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        delta_list = []  # 收集上传的本地梯度字典

        for i, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            client_params = get_parameters(selected_model)
            theta_list.append(client_params)
            delta_list.append(results[i]['local_delta'])

            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        weights = [client_datasets_size_list[j] for j in idxs_users]
        theta_avg = np.average(theta_list, axis=0, weights=weights).tolist()
        set_parameters(global_model, theta_avg)

        # ===== FedLGD 核心：服务器端梯度匹配（Local-Global Distillation）=====
        # 计算聚合后的全局 delta（作为目标梯度）
        if lgd_lambda > 0 and len(idxs_users) > 0:
            avg_delta = {}
            total_w = sum(weights)
            for i, delta in enumerate(delta_list):
                w = weights[i] / total_w
                for name, arr in delta.items():
                    t = torch.tensor(arr) * w
                    if name not in avg_delta:
                        avg_delta[name] = t
                    else:
                        avg_delta[name] += t

            # 构造虚拟输入（根据任务类型）
            global_model.to(device)
            global_model.train()
            server_optimizer = BERTCLF_Optimizer(
                method="SGD", learning_rate=lgd_lr, max_grad_norm=0)
            server_optimizer.set_parameters(list(global_model.named_parameters()))

            emb_dim = param_dict.get('emb_dim', 768 if "SENT_CLF" in param_dict["task"] else
                                      (512 if "IMG_CLF" in param_dict["task"] else param_dict.get('nn_input_size', 128)))

            gpu_s = time.time()
            for _step in range(lgd_steps):
                if "SENT_CLF" in param_dict["task"]:
                    max_len = param_dict.get('max_len', 128)
                    input_ids = torch.randint(0, 30522, (num_virtual, max_len), device=device)
                    attention_mask = torch.ones((num_virtual, max_len), device=device)
                    labels_v = torch.randint(0, 2, (num_virtual,), device=device)
                    features, logits = global_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss_task = torch.sum(criterion(logits, labels_v)) / num_virtual
                elif "IMG_CLF" in param_dict["task"]:
                    # 随机噪声图像
                    inp_ch = 3 if param_dict.get('dataset', '').lower() not in ['fmnist', 'mnist'] else 1
                    imgs = torch.randn(num_virtual, inp_ch, 32, 32, device=device)
                    labels_v = torch.round(torch.rand(num_virtual, device=device)).long()
                    preds, features = global_model(imgs)
                    loss_task = torch.sum(criterion(preds[:, 0], labels_v.float())) / num_virtual
                else:  # Tabular_CLF
                    inp_size = param_dict.get('nn_input_size', 128)
                    X = torch.randn(num_virtual, inp_size, device=device)
                    labels_v = torch.round(torch.rand(num_virtual, device=device)).long()
                    if "ANN" in str(type(global_model)):
                        preds, features = global_model(X)
                    else:
                        preds = global_model(X)
                        features = X
                    loss_task = torch.sum(criterion(preds[:, 0], labels_v.float())) / num_virtual

                # 计算当前模型参数关于虚拟样本的梯度
                trainable = [p for p in global_model.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss_task, trainable, create_graph=False, allow_unused=True)

                # 梯度匹配损失：使当前梯度方向与聚合梯度方向一致（L2）
                loss_match = 0.0
                for (name, p), g in zip(global_model.named_parameters(), grads):
                    if g is None or name not in avg_delta:
                        continue
                    target = avg_delta[name].to(device)
                    loss_match += torch.norm(g.view(-1) - target.view(-1).detach(), p=2) ** 2

                loss_total = loss_task + lgd_lambda * loss_match
                server_optimizer.zero_grad()
                loss_total.backward()
                server_optimizer.step()

            gpu_seconds_match = time.time() - gpu_s
            total_gpu_seconds += gpu_seconds_match
            global_model.to("cpu")

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(f"Testing round {iter_t + 1}/{communication_round_I}")
        logger.info(f"Total GPU seconds: {total_gpu_seconds:.1f}, Avg: {avg_gpu_seconds:.1f}")

        del theta_list
        gc.collect()

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
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "Tabular_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*2*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()

        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t + 1, client_model_updates=client_model_updates)

        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(param_dict=param_dict, iter_t=iter_t, global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time)
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("FedLGD training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedLGD.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
