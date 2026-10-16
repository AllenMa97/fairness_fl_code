# https://arxiv.org/pdf/2109.05662
# FedFair: Fairness-Aware Federated Learning
# 核心思想：在本地训练中引入公平性约束，通过正则化项确保各敏感属性组之间的公平性
import os
import gc
import copy
import time
import torch
import numpy as np
from transformers import BertTokenizer
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from module.dataset import (MoJiDataset, BiosDataset, MTCDataset, CelebaDataset,
                            get_UTKFace_dataset, get_FairFace_dataset, get_LFWAPlus_dataset)
from torch.utils.data import DataLoader, Subset
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step, AutocastSafeBCELoss
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.experiment_state import (
    AlgorithmRunResult,
    capture_training_loader_generator_states,
    restore_training_loader_generator_states,
)


def D_hat_θ(param_dict, client_dataset, client_model, device, tokenizer,
            coef_ac, coef_bc, scaler):
    # According to Eq 7., the D_hat_θ = L_hat_a,c - L_hat_b,c , where a & b are the value of the sensitive attribute.
    # 注意：下面四段损失计算必须保持可导（torch.enable_grad），否则全局模型拿不到梯度
    # （原实现的 torch.no_grad 使目标函数与全局模型脱钩，全局模型冻结在初值）。
    # 梯度采用「逐批反传」：每个批次算完立刻 backward 并只保留标量（见下方 _accumulate）。
    # 若整份客户端数据集累加后再统一反传，全部批次的前向运算图会常驻显存。
    # 四个 (s,c) 组的线性系数不同：s=a 的两组（L_hat_ac0/ac1）系数为 coef_ac，
    # s=b 的两组（L_hat_bc0/bc1）系数为 coef_bc，二者由客户端损失的线性展开得到。
    # 显存优化：四个组的 DataLoader 用 64 的 micro-batch（原 256）。梯度本身已经在
    # 组内逐批累加到全局模型（不逐步 step），缩小单批只降低峰值激活显存，对
    # 组内总损失（按组样本数归一的和）无影响。
    #
    # ===== 显存踩坑实测记录（勿删，供后续排障对照）=====
    # 历史峰值（batch=256/FP32，图像数据集）：
    # - UTKFace / FairFace 单进程实测峰值 41.42 GiB，触发 CUDA OOM。
    # - 修复顺序与结论：先「整份数据集累加后反传」改为逐批反传（41.42 GiB 的主要
    #   持有者之一是全部批次的前向计算图）；逐批反传后峰值仍偏高，再把单批
    #   batch=256 缩为 micro-batch=64（峰值激活与单批大小成正比）。两步缺一不可。
    # - 表格数据集（ADULT/COMPAS/DRUG/DUTCH）与文本数据集显存占用低（数 GiB 级），
    #   不构成压力；高危场景集中在图像 + batch=256。
    # - OOM 时先看报错里的 "this process has X GiB"（本进程实测），旁边的大数字
    #   进程可能是同卡 FedFair/FL_FairBatch 邻居或常驻 VLLM::Worker（约 42 GiB），
    #   排卡按「同卡峰值之和」估算，勿只看单进程。
    micro_batch_size = max(1, param_dict['batch_size'] // 4)

    # ===== 客户端数据必须按 Subset.indices 取分片（勿改回 client_dataset.dataset.*）=====
    # client_dataset 由 module.dataloader._client_subsets 构造成 Subset，
    # Subset.dataset 会脱掉索引层、返回「整份训练集」，.indices 才是本客户端的行号。
    # 旧实现直接读 client_dataset.dataset.X/.labels/.protected，于是所有客户端、
    # 所有划分的四个 (s,c) 分组都取自整份训练集，D_hat_θ 与「客户端」「划分」双双脱钩，
    # 表现为同一算法跨划分逐字节相同（这正是 FedFair 表格结果不可用的原因）。
    full_dataset = client_dataset.dataset if isinstance(client_dataset, Subset) else client_dataset
    if isinstance(client_dataset, Subset):
        client_indices = np.asarray(client_dataset.indices, dtype=np.int64)
    else:
        client_indices = np.arange(len(full_dataset.labels), dtype=np.int64)

    if "SENT_CLF" in param_dict["task"]:
        client_X = [full_dataset.texts[int(i)] for i in client_indices]
    else:
        client_X = None

    client_y = np.asarray(full_dataset.labels)[client_indices]
    try:
        client_s = np.asarray(full_dataset.protected)[client_indices]
    except Exception:
        client_s = np.asarray(full_dataset.s1)[client_indices]

    a, b = 1, 0

    c0, c1 = (client_y == 0), (client_y == 1)
    sa, sb = (client_s == a), (client_s == b)

    sa_c0, sa_c1 = sa * c0, sa * c1
    sb_c0, sb_c1 = sb * c0, sb * c1

    m_sa_c0, m_sa_c1 = sum(sa_c0), sum(sa_c1)
    m_sb_c0, m_sb_c1 = sum(sb_c0), sum(sb_c1)

    X_sa_c0, X_sa_c1, y_sa_c0, y_sa_c1 = [], [], [], []

    X_sb_c0, X_sb_c1, y_sb_c0, y_sb_c1 = [], [], [], []

    # 构建四个 (s,c) 组的数据引用：
    #   SENT_CLF      -> X_* 存文本本身（Dataset_func 直接吃 texts/labels）
    #   IMG_CLF/Tabular_CLF -> X_* 存「整份训练集上的全局行号」，
    #                          后面用 Subset(full_dataset, X_*) 取数据，必须用全局行号。
    for index in range(len(client_y)):
        data_ref = client_X[index] if "SENT_CLF" in param_dict["task"] else int(client_indices[index])
        flag_1 = sa_c0[index]
        flag_2 = sa_c1[index]
        flag_3 = sb_c0[index]
        flag_4 = sb_c1[index]

        if flag_1:
            X_sa_c0.append(data_ref)
            y_sa_c0.append(client_y[index])
        if flag_2:
            X_sa_c1.append(data_ref)
            y_sa_c1.append(client_y[index])
        if flag_3:
            X_sb_c0.append(data_ref)
            y_sb_c0.append(client_y[index])
        if flag_4:
            X_sb_c1.append(data_ref)
            y_sb_c1.append(client_y[index])

    # 四种数据分别计算损失
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]) or "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)

    dataset_name = param_dict['dataset_name'].lower()
    if "moji".lower() in dataset_name:
        Dataset_func = MoJiDataset
    elif "bios".lower() in dataset_name:
        Dataset_func = BiosDataset
    elif "mtc".lower() in dataset_name:
        Dataset_func = MTCDataset
    elif "celeba".lower() in dataset_name:
        Dataset_func = CelebaDataset

    # 逐批反传：把该批次损失乘以对应的线性系数后立刻 backward，只保留标量。
    # client_i_loss = (n_i/N)·L + (λ_a-λ_b)/K·D = coef_ac·L_ac + coef_bc·L_bc，
    # 其中 L 与 D 都是各批次损失之和，故「和的梯度 = 梯度之和」，逐批反传与整份
    # 数据集一次性反传数学等价，但峰值显存从「全部批次」降到「单个批次」。
    def _accumulate(term, coef):
        scale_backward(coef * term, scaler)
        return float(term.detach())

    # testing_dataset / testloader 只在某个分组非空时才会被赋值，
    # 这里先置空以保证末尾的 del 一定合法（否则四个分组全空会抛 NameError）。
    testing_dataset, testloader = None, None

    # 第一种数据,s=a, c=0
    # logger.info("第一种数据,s=a, c=0")
    L_hat_ac0 = 0
    if len(X_sa_c0) != 0:
        with torch.enable_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sa_c0,
                    labels=y_sa_c0,
                    protected=[a for _ in X_sa_c0],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]) or "Tabular_CLF" in param_dict["task"]:
                testing_dataset = Subset(full_dataset, X_sa_c0)
            testloader = DataLoader(testing_dataset, batch_size=micro_batch_size, shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)
                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]):
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_ac0 += _accumulate(torch.sum(batch_loss) / m_sa_c0, coef_ac)

    # 第二种数据,s=a, c=1
    # logger.info("第二种数据,s=a, c=1")
    L_hat_ac1 = 0
    if len(X_sa_c1) != 0:
        with torch.enable_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sa_c1,
                    labels=y_sa_c1,
                    protected=[a for _ in X_sa_c1],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]) or "Tabular_CLF" in param_dict["task"]:
                testing_dataset = Subset(full_dataset, X_sa_c1)
            testloader = DataLoader(testing_dataset, batch_size=micro_batch_size, shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)

                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]):
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_ac1 += _accumulate(torch.sum(batch_loss) / m_sa_c1, coef_ac)

    # 第三种数据,s=b, c=0
    # logger.info("第三种数据,s=b, c=0")
    L_hat_bc0 = 0
    if len(X_sb_c0) != 0:
        with torch.enable_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sb_c0,
                    labels=y_sb_c0,
                    protected=[a for _ in X_sb_c0],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]) or "Tabular_CLF" in param_dict["task"]:
                testing_dataset = Subset(full_dataset, X_sb_c0)
            testloader = DataLoader(testing_dataset, batch_size=micro_batch_size, shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)
                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]):
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_bc0 += _accumulate(torch.sum(batch_loss) / m_sb_c0, coef_bc)


    # 第四种数据,s=b, c=1
    # logger.info("第四种数据,s=b, c=1")
    L_hat_bc1 = 0
    if len(X_sb_c1) != 0:
        with torch.enable_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sb_c1,
                    labels=y_sb_c1,
                    protected=[a for _ in X_sb_c1],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]) or "Tabular_CLF" in param_dict["task"]:
                testing_dataset = Subset(full_dataset, X_sb_c1)
            testloader = DataLoader(testing_dataset, batch_size=micro_batch_size, shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)
                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]):
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_bc1 += _accumulate(torch.sum(batch_loss) / m_sb_c1, coef_bc)

    del testing_dataset, testloader
    gc.collect()
    torch.cuda.empty_cache()

    L_hat_ac = L_hat_ac0 + L_hat_ac1
    L_hat_bc = L_hat_bc0 + L_hat_bc1

    return L_hat_ac + L_hat_bc, L_hat_ac - L_hat_bc


def _train_single_client_fedfair(client_id, device, model, param_dict, training_dataloaders,
                                  algorithm_epoch_T, client_datasets_size_list, client_datasets_total_size,
                                  num_clients_K, tokenizer, iter_t, communication_round_I,
                                  λ_a, λ_b, β, γ, scaler):
    """FedFair 单客户端训练函数"""
    logger.info("Copy From Global Model")
    model.train()
    model.to(device)

    client_loss_list = []
    client_D_hat_list = []

    # 线性系数在客户端训练前即已确定（λ 在每轮训练结束后才更新）：
    # client_i_loss = (n_i/N)·L + (λ_a-λ_b)/K·D = coef_ac·L_ac + coef_bc·L_bc。
    # 传入 D_hat_θ 后由其逐批反传，梯度直接累加到全局模型上，此处不再延迟反传。
    client_weight = client_datasets_size_list[client_id] / client_datasets_total_size
    fair_coef = (λ_a - λ_b) / num_clients_K
    coef_ac = client_weight + fair_coef
    coef_bc = client_weight - fair_coef

    for epoch in range(algorithm_epoch_T):
        L_hat_all, D_hat_i_θ = D_hat_θ(param_dict, training_dataloaders[client_id].dataset, model, device,
                                       tokenizer, coef_ac, coef_bc, scaler)
        client_i_loss = client_weight * L_hat_all + fair_coef * D_hat_i_θ
        client_loss_list.append(client_i_loss)
        client_D_hat_list.append(D_hat_i_θ)
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {client_i_loss / client_datasets_size_list[client_id]}")

    gc.collect()
    torch.cuda.empty_cache()

    return {'client_loss_list': client_loss_list, 'client_D_hat_list': client_D_hat_list}


# FedFair直接训练的是全局模型

def FedFair(device,
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


    # BERT tokenizer 仅在 SENT_CLF 任务下被使用（见 D_hat_θ 与 _train_single_client_fedfair）。
    # 表格/图像任务不构建 tokenizer：既省去无谓开销，也避免在无外网且无离线权重的环境下
    # 反复 HEAD bert-base-uncased 超时（重试 5 次后抛 OSError），导致该算法每个数据集
    # 连续失败 5 次并被调度器放弃。SENT_CLF 下仍走离线优先解析，与 (experiment_setup) 一致。
    tokenizer = None
    if "SENT_CLF" in param_dict["task"]:
        from module.experiment_setup import _resolve_bert_model_path
        tokenizer = BertTokenizer.from_pretrained(
            _resolve_bert_model_path(param_dict.get("bert_model_name_or_path"))
        )

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)
    client_datasets_total_size = sum(client_datasets_size_list)


    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]) or "Tabular_CLF" in param_dict["task"]:
        criterion = AutocastSafeBCELoss(reduction='none').to(device)

    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(global_model.named_parameters()))

    # 断点恢复：恢复计数器、AMP、客户端选择历史与 DataLoader 生成器状态
    total_gpu_seconds = 0.0 if resume_state is None else float(resume_state.total_gpu_seconds)
    previous_state = {} if resume_state is None else dict(resume_state.algorithm_state)
    if resume_state is not None:
        restore_training_loader_generator_states(
            training_dataloaders, previous_state.get("training_loader_generator_states")
        )
    users_gpu_seconds_list = list(previous_state.get("users_gpu_seconds_list", [0.0] * num_clients_K))
    if len(users_gpu_seconds_list) != num_clients_K:
        raise RuntimeError("FedFair checkpoint has incompatible users_gpu_seconds_list")
    total_communication_cost = 0.0 if resume_state is None else float(resume_state.total_communication_cost)
    client_selection_history = [] if resume_state is None else [list(row) for row in resume_state.client_selection_history]
    runtime_offset = 0.0 if resume_state is None else float(resume_state.total_runtime_seconds)
    runtime_start = time.monotonic()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # FedFair超参数
    β = 0.05
    γ = 0.001
    ϵ = 0.05
    # 拉格朗日乘子 λ_a, λ_b 为跨轮维护状态，续跑时从上一次检查点恢复
    λ_a = float(previous_state.get("lambda_a", 0.15))
    λ_b = float(previous_state.get("lambda_b", 1.0))

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
        )
        selected_clients = idxs_users.tolist() if hasattr(idxs_users, "tolist") else list(idxs_users)
        client_selection_history.append([int(client_id) for client_id in selected_clients])
        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        client_loss_list = []
        client_D_hat_list = []

        # 记录GPU计算开始时间
        gpu_start_time = time.time()

        # Simulate Client Parallel
        # FedFair 的客户端损失必须把梯度回传到全局模型（Eq 10 的全局更新依赖它），
        # 因此不能走 ClientParallelExecutor 的模型副本：副本上的计算图与全局模型参数
        # 无关联，反传只会落在副本上，全局模型依旧拿不到梯度。这里改为在全局模型
        # 本身上串行执行客户端；梯度在 D_hat_θ 内部逐批反传并累加，客户端之间不重置，
        # 即等价于对所有选中客户端损失求和后一次性反传，但峰值显存仅为一个批次。
        for client_id in idxs_users:
            client_result = _train_single_client_fedfair(
                client_id, device, global_model,
                param_dict=param_dict,
                training_dataloaders=training_dataloaders,
                algorithm_epoch_T=algorithm_epoch_T,
                client_datasets_size_list=client_datasets_size_list,
                client_datasets_total_size=client_datasets_total_size,
                num_clients_K=num_clients_K,
                tokenizer=tokenizer,
                iter_t=iter_t,
                communication_round_I=communication_round_I,
                λ_a=λ_a, λ_b=λ_b, β=β, γ=γ,
                scaler=scaler,
            )
            client_D_hat_list.extend(client_result['client_D_hat_list'])
            # 梯度已在 D_hat_θ 内逐批反传完毕，此处只收集标量用于日志与乘子更新
            client_loss_list.extend(float(loss) for loss in client_result['client_loss_list'])
            del client_result
            gc.collect()
            torch.cuda.empty_cache()

        # 再走一个训练batch构建运算图
        tmp_loss = 0
        client_i_dataloader = training_dataloaders[idxs_users[0]]
        global_model.train()
        global_model.to(device)
        for batch_id, batch in enumerate(client_i_dataloader):
            # labels尺寸 [batch_size]
            labels = batch["labels"][:2].to(device)
            with autocast_context(device, use_amp):
                if "SENT_CLF" in param_dict["task"]:
                    input_ids = batch["input_ids"][:2].to(device)
                    attention_mask = batch["attention_mask"][:2].to(device)

                    features, logits = global_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    activated_preds = logits
                    _, preds = torch.max(activated_preds, dim=1)
                    tmp_loss += criterion(activated_preds, labels)
                    del input_ids, attention_mask, labels, features, logits, activated_preds, _, preds
                elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]):
                    imgs = batch["img"][:2].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = global_model(imgs)
                    tmp_loss += criterion(preds[:, 0], labels.float())
                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"][:2].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(global_model)):
                        local_prediction, features = global_model(X)
                    elif "LogisticRegression" in str(type(global_model)):
                        local_prediction = global_model(X)
                    else:
                        local_prediction = global_model(X)
                    tmp_loss = criterion(local_prediction[:, 0], labels.float())

            gc.collect()
            torch.cuda.empty_cache()
            # 这里一定要break，因为仅仅只是为了构造运算图
            break

        # Parameter update by Equation 10
        # 客户端梯度已在上面的循环中逐客户端累加完毕；tmp_loss 仅用于保证全局模型
        # 存在一张可反传的计算图（与原始实现一致，系数为 0，不产生实际梯度）。
        global_loss = 0*tmp_loss.sum() + sum(client_loss_list)

        scale_backward(global_loss, scaler)
        scaler_step(scaler, optimizer)
        optimizer.zero_grad()

        client_model_updates = []

        # 记录GPU计算结束时间
        gpu_end_time = time.time()

        common_term_of_the_first_term_in_Eq12_Eq13 = 1 - γ * β
        first_term_in_Eq12 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_a
        first_term_in_Eq13 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_b

        accumulation_of_client_D_hat_list = float(sum(client_D_hat_list))
        second_term_in_Eq12_Eq_13 = (β / len(idxs_users)) * accumulation_of_client_D_hat_list

        third_term_in_Eq12_Eq_13 = β * ϵ

        eq_12 = first_term_in_Eq12 + second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13
        eq_13 = first_term_in_Eq13 - second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13
        # 更新FedFair引入的参数
        λ_a = max(eq_12, 0)
        λ_b = max(eq_13, 0)

        total_gpu_seconds += (gpu_end_time - gpu_start_time)
        total_communication_cost += len(idxs_users) * 2 * model_MB_size
        avg_gpu_seconds = total_gpu_seconds / num_clients_K

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


            elif ("IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]):
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

        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

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
                    "lambda_a": λ_a,
                    "lambda_b": λ_b,
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
    save_path = os.path.join(save_dir, f"global_FedFair.pt")
    torch.save(global_model, save_path)
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state={
            "users_gpu_seconds_list": users_gpu_seconds_list,
            "training_loader_generator_states": capture_training_loader_generator_states(training_dataloaders),
            "lambda_a": λ_a,
            "lambda_b": λ_b,
        },
        amp_scaler_state=None if scaler is None else scaler.state_dict(),
        client_selection_history=client_selection_history,
    )
