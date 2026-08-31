# Fair-FedMOE: Group-Fair One-Shot Federated Learning via Prototype-Guided Experts for Medical Imaging Analysis
# ICML 2026 (Poster) | https://github.com/lylmlz/Fair-FedMOE
# 核心思想: 面向群组公平的联邦 MoE。本地训练阶段通过 Fairness-aware Expert Routing：可学习的群组
#           原型将样本路由到组专属专家（Top-K 余弦相似度路由），使各子群学习组特定特征、避免组间干扰，
#           损失 = CE + 原型对比损失 + 原型多样性正则 + 初始符号掩码正则 + 专家头 L1 稀疏。
#           聚合阶段通过 Prototype-guided Differential Aggregation：以各客户端原型（展平后）的余弦
#           相似度混合数据量权重得到个性化权重矩阵，骨干参数做符号一致性加权的差异聚合（过滤冲突
#           更新），专家头做稀疏重叠聚合，得到每个客户端的个性化模型。
# Core Idea: Group-fair federated MoE. Local stage (Fairness-aware Expert Routing): learnable group
#            prototypes route samples to group-specific experts via Top-K cosine similarity so that
#            subgroups specialize without inter-group interference; loss = CE + prototype contrastive
#            + prototype diversity reg + initial-sign-mask reg + expert-head L1. Aggregation stage
#            (Prototype-guided Differential Aggregation): a personalized weight matrix mixing prototype
#            cosine similarity with data-size FedAvg weights; sign-consistency masked delta aggregation
#            for the backbone (filters conflicting updates) and sparsity-overlap aggregation for heads.
# 框架适配说明 / Adaptation notes:
#   1. 原论文基于 ViT 基础模型并替换分类头；本框架骨干异构（BERT/CNN/ANN），故以"骨干特征 -> 投影 ->
#      原型路由 -> 组专家"的原型 MoE 头包装框架模型（wrapper），不改框架模型定义。
#   2. 原论文为 one-shot 且输出 N 个个性化模型；本框架为多轮单全局模型，故每轮执行
#      "个性化差异聚合 -> 数据量加权平均得到单一全局模型"。
#   3. 原论文的原型对比损失 SupConLossWithPrototypes 未开源完整实现，此处采用 ProtoNCE 风格：
#      样本投影特征与其群组原型拉近、与其余群组原型推远（温度 T）。
#   4. 原论文初始化原型用 KMeans(1)（即组内特征均值）；本实现在第 1 轮由客户端本地计算组内投影特征
#      均值完成初始化（数据不出本地），后续轮次沿用服务器聚合的原型。

import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool.logger import *
from tool.utils import set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.tensorboard_logger import log_test_metrics, log_system_metrics, flush, log_deep_metrics, get_monitoring_config
from tool.client_parallel import ClientParallelExecutor

HEAD_KEYWORD = "attribute_experts"   # 专家头参数命名关键字（L1 与稀疏聚合均以此划分）


class ANN_FairFedMOE(nn.Module):
    """原型引导 MoE 包装模型 / Prototype-guided MoE wrapper around a framework base model.

    命名说明: 类名包含 "ANN" 是为了兼容框架 Tabular 测试函数的类型分发
    （tool/utils.py 中 "ANN" in str(type(model)) 判断），并非限定表格任务。
    """

    def __init__(self, base_model, task, feat_dim, out_dim, num_groups,
                 reduced_dim, temperature, top_k):
        super().__init__()
        self.base_model = base_model
        self.task = task
        self.num_groups = num_groups
        self.temperature = temperature
        self.top_k = min(top_k, num_groups)
        self.reduced_dim = reduced_dim

        # 特征投影（原论文 proj: Linear-GELU-Linear）
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, reduced_dim),
        )
        # 可学习群组原型 (num_groups, reduced_dim)
        self.attribute_prototypes = nn.Parameter(torch.randn(num_groups, reduced_dim) * 0.02)
        # 组专属专家头
        self.attribute_experts = nn.ModuleList(
            [nn.Linear(reduced_dim, out_dim) for _ in range(num_groups)])
        for expert in self.attribute_experts:
            nn.init.trunc_normal_(expert.weight, std=0.02)
            nn.init.zeros_(expert.bias)

    def __getattr__(self, name):
        # 未在 wrapper 上找到的属性回退到基础模型（保持 .bert/.shared_base 等访问兼容）
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, '_modules')
            if 'base_model' in modules:
                return getattr(modules['base_model'], name)
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")

    def moe_logits(self, features):
        """原型路由 + Top-K 专家融合 / Prototype routing + Top-K expert fusion."""
        z = self.proj(features)
        z_norm = F.normalize(z, p=2, dim=1)
        proto_norm = F.normalize(self.attribute_prototypes, p=2, dim=1)
        similarities = torch.mm(z_norm, proto_norm.t())
        k = min(self.top_k, self.num_groups)
        topk_sims, topk_indices = torch.topk(similarities, k=k, dim=1)
        top_k_weights = torch.softmax(topk_sims / self.temperature, dim=1)
        expert_outputs_all = torch.stack(
            [expert(z) for expert in self.attribute_experts], dim=1)  # (B, G, out_dim)
        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs_all.size(-1))
        selected = torch.gather(expert_outputs_all, 1, indices_expanded)
        return (selected * top_k_weights.unsqueeze(-1)).sum(dim=1), z

    def forward(self, *args, **kwargs):
        if "SENT_CLF" in self.task:
            features, _ = self.base_model(
                input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"])
            logits, _ = self.moe_logits(features)
            return features, logits
        elif "IMG_CLF" in self.task:
            imgs = args[0] if args else kwargs["img"]
            _, features = self.base_model(imgs)
            logits, z = self.moe_logits(features)
            pred = torch.sigmoid(logits).view(-1, 1)
            return pred, z
        else:  # Tabular_CLF
            X = args[0] if args else kwargs["X"]
            if "LogisticRegression" in str(type(self.base_model)):
                features = X
            else:
                _, features = self.base_model(X)
            logits, z = self.moe_logits(features)
            pred = torch.sigmoid(logits).view(-1, 1)
            return pred, z


def _probe_model_dims(base_model, param_dict, testing_dataloader, device):
    """用服务器持有的测试集首个 batch 探测骨干特征维与输出维（仅取形状，不看数据值）。"""
    base_model.to(device)
    base_model.eval()
    batch = next(iter(testing_dataloader))
    with torch.no_grad():
        if "SENT_CLF" in param_dict["task"]:
            features, logits = base_model(input_ids=batch["input_ids"].to(device),
                                          attention_mask=batch["attention_mask"].to(device))
            return features.shape[1], logits.shape[1]
        elif "IMG_CLF" in param_dict["task"]:
            _, features = base_model(batch["img"].to(device))
            return features.shape[1], 1
        else:
            X = batch["X"].to(device)
            if "LogisticRegression" in str(type(base_model)):
                out = base_model(X)
                features = X
            else:
                out, features = base_model(X)
            return features.shape[1], out.shape[1]


def _supcon_prototype_loss(z, prototypes, groups, temperature):
    """ProtoNCE 风格原型对比损失：拉近样本与其群组原型，推远其他群组原型。"""
    z_norm = F.normalize(z, p=2, dim=1)
    proto_norm = F.normalize(prototypes, p=2, dim=1)
    sims = torch.mm(z_norm, proto_norm.t()) / temperature  # (B, G)
    targets = groups.long()
    valid = (targets >= 0) & (targets < prototypes.size(0))
    if valid.sum() == 0:
        return z.new_tensor(0.0)
    return F.cross_entropy(sims[valid], targets[valid])


def _prototype_diversity_loss(prototypes):
    """原型多样性正则：组间余弦相似度尽量接近单位阵（与原论文 compute_prototype_regularization 一致）。"""
    proto_norm = F.normalize(prototypes, p=2, dim=1)
    sim_matrix = torch.mm(proto_norm, proto_norm.t())
    identity = torch.eye(prototypes.size(0), device=prototypes.device)
    return torch.mean((sim_matrix - identity) ** 2)


def _init_prototypes_locally(model, dataloader, param_dict, device, max_batches=8):
    """第 1 轮客户端本地原型初始化：组内投影特征均值（数据不出本地）。"""
    model.eval()
    feat_sums = torch.zeros(model.num_groups, model.reduced_dim, device=device)
    counts = torch.zeros(model.num_groups, device=device)
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if b_idx >= max_batches:
                break
            if "SENT_CLF" in param_dict["task"]:
                features, _ = model.base_model(input_ids=batch["input_ids"].to(device),
                                               attention_mask=batch["attention_mask"].to(device))
            elif "IMG_CLF" in param_dict["task"]:
                _, features = model.base_model(batch["img"].to(device))
            else:
                X = batch["X"].to(device)
                if "LogisticRegression" in str(type(model.base_model)):
                    features = X
                else:
                    _, features = model.base_model(X)
            z = model.proj(features)
            groups = batch["protected"].to(device).long()
            for g in range(model.num_groups):
                mask = groups == g
                if mask.sum() > 0:
                    feat_sums[g] += z[mask].sum(dim=0)
                    counts[g] += mask.sum()
    new_protos = model.attribute_prototypes.data.clone()
    for g in range(model.num_groups):
        if counts[g] > 0:
            new_protos[g] = feat_sums[g] / counts[g]
    model.attribute_prototypes.data = new_protos
    model.train()


def _train_single_client_fairfedmoe(client_id, device, model, param_dict,
                                    training_dataloaders, algorithm_epoch_T,
                                    accumulation_steps, use_amp, scaler, criterion,
                                    basic_path, iter_t, communication_round_I, num_clients_K,
                                    contrast_w, proto_reg_w, lambda_sign, lambda_l1,
                                    init_max_batches):
    """Fair-FedMOE 单客户端：本地原型初始化（首轮）+ 原型路由 MoE 本地训练（含 4 项正则）。"""
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[client_id]

    # 初始符号掩码（接收全局模型后记录，训练中惩罚违背初始符号的参数）
    initial_sign_masks = {name: torch.sign(p.detach().cpu())
                          for name, p in model.named_parameters() if p.requires_grad}

    # 首轮：本地组内特征均值初始化原型（对应原论文 initialize_prototypes）
    if iter_t == 0:
        _init_prototypes_locally(model, client_i_dataloader, param_dict, device, init_max_batches)

    gpu_seconds = 0

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0
        epoch_total_size = 0
        for batch_id, batch in enumerate(client_i_dataloader):
            gpu_start_time = time.time()
            with autocast_context(device, use_amp):
                if "SENT_CLF" in param_dict["task"]:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    groups = batch["protected"].to(device)
                    features, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_loss = criterion(logits, labels)
                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                    labels = batch["labels"].to(device)
                    groups = batch["protected"].to(device)
                    preds, z = model(imgs)
                    features = z
                    cls_loss = criterion(preds[:, 0], labels.float())
                else:
                    X = batch["X"].to(device)
                    labels = batch["labels"].to(device)
                    groups = batch["protected"].to(device)
                    preds, z = model(X)
                    features = z
                    cls_loss = criterion(preds[:, 0], labels.float())

                true_batch_size = labels.size(0)
                contrast_loss = _supcon_prototype_loss(
                    features, model.attribute_prototypes, groups, model.temperature)
                div_loss = _prototype_diversity_loss(model.attribute_prototypes)

                sign_loss = z.new_tensor(0.0)
                if lambda_sign > 0:
                    for name, p in model.named_parameters():
                        if name in initial_sign_masks:
                            mask = initial_sign_masks[name].to(p.device)
                            sign_loss = sign_loss + torch.sum(torch.relu(-mask * p))

                l1_loss = z.new_tensor(0.0)
                if lambda_l1 > 0:
                    for name, p in model.named_parameters():
                        if HEAD_KEYWORD in name and 'bias' not in name and p.requires_grad:
                            l1_loss = l1_loss + torch.sum(torch.abs(p))

                loss = (cls_loss.sum() + contrast_w * contrast_loss + proto_reg_w * div_loss
                        + lambda_sign * sign_loss + lambda_l1 * l1_loss) / true_batch_size

            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)
                model.zero_grad()
            gpu_seconds += time.time() - gpu_start_time
            epoch_total_loss += loss.item() * true_batch_size
            epoch_total_size += true_batch_size
            del features, labels, groups
            gc.collect()
        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()
        logger.info(f"Round {iter_t + 1}/{communication_round_I}; Client {client_id}/{num_clients_K}; "
                    f"Epoch {epoch + 1}; Avg Loss: {epoch_total_loss / max(epoch_total_size, 1):.4f}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)
    return {'gpu_seconds': gpu_seconds}


def _compute_similarity_weights(client_protos_list, client_data_counts, temperature, alpha):
    """原型相似度个性化权重矩阵（原论文 compute_pairwise_similarity_weights）。"""
    num_clients = len(client_protos_list)
    data_counts = torch.tensor(client_data_counts, dtype=torch.float32)
    fedavg_weights = data_counts / data_counts.sum()
    stacked = torch.stack([p.flatten() for p in client_protos_list])  # (N, G*reduced)
    norm_protos = F.normalize(stacked, p=2, dim=1)
    sim_logits = torch.mm(norm_protos, norm_protos.t())
    sim_logits = torch.nan_to_num(sim_logits, nan=0.0)
    similarity_weights = F.softmax(sim_logits / temperature, dim=1)
    return alpha * similarity_weights + (1 - alpha) * fedavg_weights.unsqueeze(0)


def _differential_aggregation(global_params, client_params_list, weights_matrix, data_weights,
                              sparsity_threshold):
    """
    Prototype-guided Differential Aggregation（原论文 OPFL_aggregation 的多轮单全局模型适配）:
    骨干: 符号一致性加权的差异聚合（过滤与自身更新方向冲突的客户端）;
    MoE 头（投影/原型/专家）: 稀疏重叠聚合（仅聚合双方均非稀疏的坐标）;
    最终以数据量加权平均 N 个个性化模型得到单一全局模型。
    """
    num_clients = len(client_params_list)
    head_keys = [k for k in global_params.keys()
                 if ("attribute_experts" in k) or ("proj." in k) or ("attribute_prototypes" in k)]
    backbone_keys = [k for k in global_params.keys() if k not in head_keys]

    W = weights_matrix.numpy()
    n = np.asarray(data_weights, dtype=np.float64)
    n = n / n.sum()

    final_params = {}
    # ---- 骨干: 符号一致性差异聚合，随后对个性化模型做数据量加权平均 ----
    for key in backbone_keys:
        g = np.asarray(global_params[key], dtype=np.float64)
        deltas = np.stack([np.asarray(cp[key], dtype=np.float64) - g for cp in client_params_list], axis=0)
        signs = np.sign(deltas)
        num_acc = np.zeros_like(g)
        den_acc = np.zeros_like(g)
        for i in range(num_clients):
            mask = (signs == signs[i:i + 1]).astype(np.float64)  # (N, *shape)
            w = W[i].reshape(-1, *([1] * (deltas.ndim - 1)))
            wm = w * mask
            num_acc += n[i] * (wm * deltas).sum(axis=0)
            den_acc += n[i] * wm.sum(axis=0)
        final_params[key] = g + num_acc / (den_acc + 1e-8)

    # ---- 专家头: 稀疏重叠聚合，随后数据量加权平均 ----
    for key in head_keys:
        stack = np.stack([np.asarray(cp[key], dtype=np.float64) for cp in client_params_list], axis=0)
        valid = (np.abs(stack) > sparsity_threshold).astype(np.float64)  # (N, *shape)
        num_acc = np.zeros_like(stack[0])
        den_acc = np.zeros_like(stack[0])
        for i in range(num_clients):
            w = W[i].reshape(-1, *([1] * (stack.ndim - 1)))
            overlap = valid[i:i + 1] * valid  # (N, *shape)
            wm = w * overlap
            num_acc += n[i] * (wm * stack).sum(axis=0)
            den_acc += n[i] * wm.sum(axis=0)
        covered = den_acc > 1e-8
        avg_pers = np.where(covered, num_acc / (den_acc + 1e-8), stack.mean(axis=0))
        final_params[key] = avg_pers

    return [final_params[k] for k in global_params.keys()]


def Fair_FedMOE(device,
                global_model,
                algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                training_dataloaders,
                training_dataset,
                client_dataset_list,
                param_dict,
                testing_dataloader,
                testing_dataset_len,
                start_round=0):
    # ---- 探测骨干特征维/输出维并构建原型 MoE 包装模型 ----
    feat_dim, out_dim = _probe_model_dims(global_model, param_dict, testing_dataloader, device)
    num_groups = int(param_dict.get('FairFedMOE_num_groups', 2))
    reduced_dim = int(param_dict.get('FairFedMOE_reduced_dim', 64))
    moe_temperature = float(param_dict.get('FairFedMOE_temperature', 0.1))
    moe_top_k = int(param_dict.get('FairFedMOE_top_k', 1))
    global_model = ANN_FairFedMOE(global_model, param_dict["task"], feat_dim, out_dim,
                                  num_groups, reduced_dim, moe_temperature, moe_top_k)

    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

    training_dataset_size = len(training_dataset.labels) if hasattr(training_dataset, 'labels') else len(training_dataset)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    del training_dataset, client_dataset_list
    gc.collect()

    # Fair-FedMOE 超参数
    contrast_w = float(param_dict.get('FairFedMOE_contrast_loss_weight', 0.1))
    proto_reg_w = float(param_dict.get('FairFedMOE_proto_reg_weight', 0.01))
    lambda_sign = float(param_dict.get('FairFedMOE_lambda_sign', 1e-4))
    lambda_l1 = float(param_dict.get('FairFedMOE_lambda_l1', 1e-4))
    agg_alpha = float(param_dict.get('FairFedMOE_agg_alpha', 0.7))
    agg_temperature = float(param_dict.get('FairFedMOE_agg_temperature', 0.1))
    sparsity_threshold = float(param_dict.get('FairFedMOE_sparsity_threshold', 1e-4))
    init_max_batches = int(param_dict.get('FairFedMOE_init_max_batches', 8))

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

        logger.info(f"Round {iter_t + 1}; Select clients: {idxs_users}; Start Local Training (Fair-FedMOE)")

        results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fairfedmoe,
            param_dict=param_dict, training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T, accumulation_steps=accumulation_steps,
            use_amp=use_amp, scaler=scaler, criterion=criterion, basic_path=basic_path,
            iter_t=iter_t, communication_round_I=communication_round_I, num_clients_K=num_clients_K,
            contrast_w=contrast_w, proto_reg_w=proto_reg_w, lambda_sign=lambda_sign,
            lambda_l1=lambda_l1, init_max_batches=init_max_batches)

        for i, client_id in enumerate(idxs_users):
            users_gpu_seconds_list[client_id] += results[i]['gpu_seconds']
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # ===== Fair-FedMOE 核心：原型相似度个性化权重 + 差异化聚合 =====
        logger.info("Fair-FedMOE: Prototype-guided Differential Aggregation")
        # 按 state_dict key 索引全局参数（与差异化聚合的骨干/专家头划分一致）
        global_state = {k: np.asarray(v, dtype=np.float64)
                        for k, v in global_model.state_dict().items()}

        client_params_list = []
        client_protos_list = []
        for i, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            state = {k: np.asarray(v, dtype=np.float64) for k, v in selected_model.state_dict().items()}
            client_params_list.append(state)
            client_protos_list.append(
                selected_model.attribute_prototypes.detach().cpu().clone())
            del selected_model
            gc.collect()

        gpu_s = time.time()
        selected_sizes = [client_datasets_size_list[j] for j in idxs_users]
        weights_matrix = _compute_similarity_weights(
            client_protos_list, selected_sizes, agg_temperature, agg_alpha)
        agg_state = _differential_aggregation(
            global_state, client_params_list, weights_matrix, selected_sizes, sparsity_threshold)
        set_parameters(global_model, [agg_state[k] for k in global_model.state_dict().keys()])
        total_gpu_seconds += time.time() - gpu_s

        global_model.to("cpu")
        del client_params_list, client_protos_list, agg_state
        gc.collect()

        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "IMG_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()
            elif "Tabular_CLF" in param_dict["task"]:
                acc, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(acc, FR)
                logger.info(f"ACC: {round(float(acc),3)}, DEO: {round(float(DEO),3)}, SPD: {round(float(SPD),3)}, FR: {round(float(FR),3)}, HM: {round(float(HM),3)}")
                log_test_metrics(accuracy=float(acc), DEO=float(DEO), SPD=float(SPD), FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=(iter_t+1)*len(idxs_users)*3*model_MB_size,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
                flush()

        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, iter_t + 1)

        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            # 断点续跑时框架会将 checkpoint 状态载入"裸"基础模型再重新包装，
            # 故此处保存基础模型状态而非 wrapper 状态（MoE 头参数不进 checkpoint）。
            save_checkpoint(param_dict=param_dict, iter_t=iter_t, global_model=global_model.base_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time)
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("Fair-FedMOE training finished.")
    save_dir = './save_path/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(global_model, os.path.join(save_dir, "global_Fair_FedMOE.pt"))
    total_comm_cost = communication_round_I * num_clients_K * FL_fraction * 3 * model_MB_size
    return global_model, total_gpu_seconds, total_comm_cost
