# FedGen: Data-Free Knowledge Distillation for Heterogeneous Federated Learning
# https://proceedings.mlr.press/v139/zhang21o.html (ICML 2021)
# 核心思想: 操作空间为轻量 latent 空间（MLP 生成器 G(z,y) → 特征向量）。
# 服务器维护一个轻量级条件生成器，根据客户端上传的分类头权重合成特征向量进行蒸馏。
# Core Idea: Operates in lightweight latent space (MLP generator G(z,y) → feature vectors).
# The server maintains a lightweight conditional generator that synthesizes feature
# vectors using client-uploaded classifier heads for knowledge distillation.

import copy
import os
import gc
import time
import torch
import torch.nn as nn
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


# ---------- FedGen 轻量级条件生成器 G(z, y) ----------
class FedGenGenerator(nn.Module):
    """MLP-based conditional generator: takes noise z + one-hot label y, outputs feature vector."""
    def __init__(self, latent_dim, num_classes, feature_dim, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z, y_onehot):
        inp = torch.cat([z, y_onehot], dim=1)
        return self.net(inp)

    def sample(self, batch_size, device):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        y = torch.randint(0, self.num_classes, (batch_size,), device=device)
        y_onehot = torch.zeros(batch_size, self.num_classes, device=device).scatter_(1, y.unsqueeze(1), 1.0)
        return self.forward(z, y_onehot), y


# ---------- 工具：提取 / 加载分类头 ----------
def _get_clf_module(model, param_dict):
    if "SENT_CLF" in param_dict["task"]:
        return model.out
    elif "IMG_CLF" in param_dict["task"]:
        return model.out_layer
    else:
        if "LogisticRegression" in str(type(model)):
            return model.layer
        return model.out_layer


def _forward_features(model, param_dict, batch, device):
    if "SENT_CLF" in param_dict["task"]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        return model.only_PLM_forward(input_ids=input_ids, attention_mask=attention_mask)
    elif "IMG_CLF" in param_dict["task"]:
        imgs = batch["img"].to(device)
        return model.only_backbone_forward(imgs)
    else:
        X = batch["X"].to(device)
        if "ANN" in str(type(model)):
            return model.only_backbone_forward(X)
        return X


def _train_single_client_fedgen(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K):
    """FedGen 单客户端训练：本地标准训练 + 上传分类头权重（服务器端蒸馏用）。"""
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
                features, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_loss = criterion(logits, labels)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
                labels = batch["labels"].to(device)
                preds, features = model(imgs)
                batch_loss = criterion(preds[:, 0], labels.float())
            else:
                X = batch["X"].to(device)
                labels = batch["labels"].to(device)
                if "ANN" in str(type(model)):
                    preds, features = model(X)
                elif "LogisticRegression" in str(type(model)):
                    preds = model(X)
                    features = X
                else:
                    preds, features = model(X) if hasattr(model, 'shared_base') else (model(X), X)
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

        avg_loss = epoch_total_loss / max(epoch_total_size, 1)
        logger.info(f"Round {iter_t+1}/{communication_round_I}; Client {client_id}/{num_clients_K}; Epoch {epoch+1}; Loss: {avg_loss:.4f}")

    # 提取分类头权重上传到服务器
    clf_module = _get_clf_module(model, param_dict)
    clf_state = {k: v.detach().cpu().numpy() for k, v in clf_module.state_dict().items()}

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    return {'gpu_seconds': gpu_seconds, 'clf_state': clf_state}


def Fed_Gen(device,
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

    # FedGen 超参数
    fg_latent_dim = int(param_dict.get('FedGen_latent_dim', 32))
    fg_hidden_dim = int(param_dict.get('FedGen_hidden_dim', 256))
    fg_steps = int(param_dict.get('FedGen_steps', 10))
    fg_batch = int(param_dict.get('FedGen_batch', 32))
    fg_lambda = float(param_dict.get('FedGen_lambda', 1.0))   # 蒸馏损失权重
    fg_gen_lr = float(param_dict.get('FedGen_gen_lr', 0.001))

    # feature dim / num classes
    if "SENT_CLF" in param_dict["task"]:
        feature_dim = param_dict.get('emb_dim', 768)
        num_classes = 2
    elif "IMG_CLF" in param_dict["task"]:
        feature_dim = param_dict.get('emb_dim', 512)
        num_classes = 2
    else:
        feature_dim = param_dict.get('nn_input_size', 128)
        num_classes = 2

    basic_path = param_dict['model_path']
    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(global_model, full_path)

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    # 初始化服务器端生成器 G(z, y)
    generator = FedGenGenerator(fg_latent_dim, num_classes, feature_dim, fg_hidden_dim).to(device)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=fg_gen_lr)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    gen_MB_size = sum(p.numel() for p in generator.parameters()) * 4 / (1024 * 1024)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parallel_executor = ClientParallelExecutor(
        device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K, fraction=FL_fraction,
            dataset_size=training_dataset_size, client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate, style="FedAvg")

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start FedGen Local Training")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fedgen,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K)

        for i, cid in enumerate(idxs_users):
            users_gpu_seconds_list[cid] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # 聚合模型
        logger.info("FedGen: Aggregate + Generator-driven distillation")
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []
        theta_list = []
        clf_states = []
        for i, cid in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            cp = get_parameters(selected_model)
            theta_list.append(cp)
            clf_states.append(results[i]['clf_state'])
            updates = {}
            for j, (p_l, p_g) in enumerate(zip(cp, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_l) - torch.tensor(p_g)
            client_model_updates.append(updates)
            del selected_model
            gc.collect()

        weights = [client_datasets_size_list[j] for j in idxs_users]
        theta_list = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_list, axis=0, weights=weights).tolist()
        set_parameters(global_model, theta_avg)

        # ===== FedGen 核心：生成器合成特征 + 分类头集成蒸馏 =====
        gpu_s = time.time()
        global_model.to(device)
        global_model.train()
        clf_modules = []
        # 从各客户端分类头权重重建临时分类头
        for state in clf_states:
            clf_ref = copy.deepcopy(_get_clf_module(global_model, param_dict))
            try:
                sd = {k: torch.tensor(v) for k, v in state.items()}
                clf_ref.load_state_dict(sd)
            except Exception:
                pass
            clf_modules.append(clf_ref.to(device).eval())

        if len(clf_modules) > 0:
            global_clf = _get_clf_module(global_model, param_dict)
            global_clf.train()
            distill_optim = BERTCLF_Optimizer(method="ADAM", learning_rate=param_dict['learning_rate']*0.1, max_grad_norm=0)
            distill_optim.set_parameters(list(global_clf.named_parameters()))

            for _step in range(fg_steps):
                # G 合成特征
                gen_optimizer.zero_grad()
                z = torch.randn(fg_batch, fg_latent_dim, device=device)
                y = torch.randint(0, num_classes, (fg_batch,), device=device)
                y_oh = torch.zeros(fg_batch, num_classes, device=device).scatter_(1, y.unsqueeze(1), 1.0)
                feats_gen = generator(z, y_oh)

                # 所有客户端分类头投票 → 集成 logits（用作 teacher）
                with torch.no_grad():
                    teacher_logits = []
                    for m in clf_modules:
                        if "SENT_CLF" in param_dict["task"]:
                            lg = m(feats_gen)
                        else:
                            lg = m(feats_gen)  # BCELoss 场景 sigmoid 已在 loss 外
                        teacher_logits.append(lg.unsqueeze(0))
                    teacher_avg = torch.cat(teacher_logits, dim=0).mean(dim=0)

                # 全局分类头 student 预测
                if "SENT_CLF" in param_dict["task"]:
                    student_logits = global_clf(feats_gen)
                    # CE 对齐 teacher 分布
                    teacher_probs = torch.softmax(teacher_avg, dim=1).detach()
                    loss_kd = -torch.sum(teacher_probs * torch.log_softmax(student_logits, dim=1), dim=1).mean()
                else:
                    student_logits = global_clf(feats_gen)
                    teacher_probs = torch.sigmoid(teacher_avg).detach()
                    bce = torch.nn.BCELoss(reduction='mean')
                    loss_kd = bce(torch.sigmoid(student_logits), teacher_probs)

                # 生成器目标：最大化分类器一致性（负 loss_kd 以训练 G）
                loss_gen = -loss_kd
                gen_optimizer.zero_grad()
                loss_gen.backward(retain_graph=True)
                gen_optimizer.step()

                # 蒸馏：更新全局分类头
                distill_optim.zero_grad()
                (fg_lambda * loss_kd).backward()
                distill_optim.step()

        total_gpu_seconds += (time.time() - gpu_s)
        global_model.to("cpu")

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(f"Testing round {iter_t+1}; GPU: {total_gpu_seconds:.1f}s, Avg: {avg_gpu_seconds:.1f}s")

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
        log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t+1, client_model_updates=client_model_updates)

        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(param_dict=param_dict, iter_t=iter_t, global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time)
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("FedGen training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_FedGen.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * (2 * model_MB_size + gen_MB_size)
    return global_model, total_gpu_seconds, total_comm_cost
