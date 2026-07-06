"""
Weights & Biases日志记录工具 (v4 - 深度监控版)
用于在联邦学习实验中记录各种指标到wandb。
参考HuggingFace TRL、字节VERL、SAM、EWC、Neural Collapse等框架与顶会论文的最佳实践。

支持的指标类别：
  - test/:               每轮通信后的测试指标
  - gradient/:           梯度监控 (norm_l2, variance, cos_sim, GSNR)
  - sample/:             样本难度监控 (loss/confidence分布, subgroup_loss)
  - embedding/:          表征质量 (intra_inter_distance, sensitive_separability)
  - fisher/:             参数重要性 (Fisher对角线, 层相对重要性) ⬅ V4 NEW
  - landscape/:          Loss landscape sharpness (SAM度量) ⬅ V4 NEW
  - update/:             模型更新统计 (稀疏度, 稳定性, 更新熵) ⬅ V4 NEW
  - activation/:         激活统计 (死神经元, 饱和度, 熵) ⬅ V4 NEW
  - neural_collapse/:    Neural Collapse指标 (类内方差, ETF偏差) ⬅ V4 NEW
  - client/:             客户端相关 (distribution JS散度) ⬅ V4 NEW
  - model/:              模型参数监控 (weight_norm, gradient_norm)
  - system/:             系统性能指标 (gpu_time, communication_cost, memory)
  - memory/:             GPU 内存监控
  - final/:              实验结束后的最终汇总指标
"""

import os
import time
import psutil
import torch
import numpy as np
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbLogger:
    def __init__(self, project=None, experiment_name=None, algorithm=None, dataset=None,
                 config=None, mode='online'):
        """
        初始化Wandb日志记录器
        
        Args:
            project: wandb项目名称
            experiment_name: 实验名称（run name）
            algorithm: 算法名称
            dataset: 数据集名称
            config: 实验配置字典
            mode: 'online'（云端同步）、'offline'（本地）或'disabled'（禁用）
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb未安装，请运行: pip install wandb")
        
        if project is None:
            project = "fairness-fl"
        
        # 构建run name
        run_name = experiment_name
        if algorithm and dataset:
            run_name = f"{algorithm}_{dataset}"
            if experiment_name:
                run_name = f"{run_name}_{experiment_name}"
        
        # 初始化wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            mode=mode,
            reinit=True
        )
        
        self.step = 0
        
        print(f"[Wandb] Project: {project}")
        print(f"[Wandb] Run: {run_name}")
        print(f"[Wandb] Mode: {mode}")
        if mode == 'online':
            print(f"[Wandb] View at: {self.run.get_url()}")

    # ──────────────────────────────────
    # 标量记录
    # ──────────────────────────────────
    def log_scalar(self, tag, value, step=None):
        if step is None:
            step = self.step
        wandb.log({tag: value}, step=step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step=None):
        if step is None:
            step = self.step
        # wandb支持直接记录多个指标
        log_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        wandb.log(log_dict, step=step)
    
    def log_metrics(self, metrics_dict, step=None, prefix=''):
        """记录一组指标，自动过滤非数值类型和NaN/Inf"""
        if step is None:
            step = self.step
        log_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                if not (np.isnan(value) or np.isinf(value)):
                    tag = f"{prefix}{key}" if prefix else key
                    log_dict[tag] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                v = value.item()
                if not (np.isnan(v) or np.isinf(v)):
                    tag = f"{prefix}{key}" if prefix else key
                    log_dict[tag] = v
        if log_dict:
            wandb.log(log_dict, step=step)

    # ──────────────────────────────────
    # 系统性能指标
    # ──────────────────────────────────
    def log_system_metrics(self, step=None, gpu_seconds=None, communication_cost=None,
                           selected_client_count=None, model_mb_size=None):
        if step is None:
            step = self.step
        metrics = {}
        if gpu_seconds is not None:
            metrics['system/gpu_seconds_total'] = float(gpu_seconds)
        if communication_cost is not None:
            metrics['system/communication_cost_mb'] = float(communication_cost)
        if selected_client_count is not None:
            metrics['system/selected_clients'] = int(selected_client_count)
        if model_mb_size is not None:
            metrics['system/model_size_mb'] = float(model_mb_size)
        self.log_metrics(metrics, step=step)
        
        # 内存使用
        try:
            wandb.log({'system/cpu_memory_pct': psutil.virtual_memory().percent}, step=step)
        except Exception:
            pass

    # ──────────────────────────────────
    # 测试指标
    # ──────────────────────────────────
    def log_test_metrics(self, accuracy=None, DEO=None, SPD=None, FR=None, HM=None,
                         step=None, gpu_seconds=None, avg_gpu_seconds=None,
                         communication_cost=None, prefix='test/'):
        """记录测试阶段的完整指标"""
        if step is None:
            step = self.step
        metrics = {}
        if accuracy is not None:
            metrics['accuracy'] = float(accuracy)
        if DEO is not None:
            metrics['DEO'] = float(DEO)
        if SPD is not None:
            metrics['SPD'] = float(SPD)
        if FR is not None:
            metrics['FR'] = float(FR)
        if HM is not None:
            metrics['HM'] = float(HM)
        if gpu_seconds is not None:
            metrics['gpu_seconds'] = float(gpu_seconds)
        if avg_gpu_seconds is not None:
            metrics['avg_gpu_seconds'] = float(avg_gpu_seconds)
        if communication_cost is not None:
            metrics['communication_cost'] = float(communication_cost)
        self.log_metrics(metrics, step=step, prefix=prefix)

    # ──────────────────────────────────
    # 客户端相关指标
    # ──────────────────────────────────
    def log_client_metrics(self, step=None, client_losses=None, client_gpu_times=None,
                           selected_clients=None, client_data_sizes=None, client_accs=None):
        """记录客户端级别的指标"""
        if step is None:
            step = self.step
        
        log_dict = {}
        if client_losses is not None:
            log_dict['client/avg_loss'] = float(np.mean(client_losses))
            wandb.log({'client/loss_distribution': wandb.Histogram(np.array(client_losses))}, step=step)
        if client_gpu_times is not None:
            log_dict['client/avg_gpu_time'] = float(np.mean(client_gpu_times))
        if selected_clients is not None:
            log_dict['client/selected_count'] = len(selected_clients)
        if client_data_sizes is not None:
            log_dict['client/avg_data_size'] = float(np.mean(client_data_sizes))
            wandb.log({'client/data_size_distribution': wandb.Histogram(np.array(client_data_sizes))}, step=step)
        if client_accs is not None:
            log_dict['client/avg_accuracy'] = float(np.mean(client_accs))
            wandb.log({'client/accuracy_distribution': wandb.Histogram(np.array(client_accs))}, step=step)
        
        if log_dict:
            wandb.log(log_dict, step=step)

    # ──────────────────────────────────
    # 模型相关指标
    # ──────────────────────────────────
    def log_model_weights(self, model, step=None):
        """记录模型权重的直方图（谨慎使用，会产生大量日志）"""
        if step is None:
            step = self.step
        log_dict = {}
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad:
                log_dict[f'model/weights/{name}'] = wandb.Histogram(param.data.cpu().numpy())
                if param.grad is not None:
                    norm = param.grad.data.norm(2).item()
                    log_dict[f'model/gradient_norm/{name}'] = norm
                total_norm += param.data.norm(2).item()
        
        log_dict['model/total_weight_norm'] = total_norm
        wandb.log(log_dict, step=step)
    
    # ──────────────────────────────────
    # V3: 梯度监控
    # ──────────────────────────────────
    def log_gradient_metrics(self, model, step=None, client_grads=None):
        """
        记录梯度相关的深度指标，全部基于客户端本地训练产生的 weight delta。
        
        数据来源：client_grads = [{param_idx: tensor_delta}, ...]
          其中 tensor_delta = θ_local - θ_global（客户端训练前后的参数变化）。
          这就是 FL 中的"有效梯度"，与 Scaffold、FedNova、Mang Ye 等工作一致。
        
        记录指标：
          - gradient/norm_l2/mean_{layer}: 各层平均更新范数（跨客户端取均值）
          - gradient/total_norm_l2: 总更新范数
          - gradient/clipping_ratio: 裁剪触发率估计
          - gradient/variance_mean: 客户端间更新方差
          - gradient/gsnr: 梯度信噪比
          - gradient/cos_sim_mean/min/distribution: 客户端间方向一致性
        """
        if step is None:
            step = self.step
        
        if client_grads is None or len(client_grads) == 0:
            return
        
        # ── 从 weight delta 推导各层梯度范数（跨客户端取均值）──
        # 收集所有客户端同一层的 delta
        layer_deltas = {}  # {param_idx: [tensor, ...]}
        for cg in client_grads:
            for name, delta in cg.items():
                if delta is not None:
                    layer_deltas.setdefault(name, []).append(delta)
        
        log_dict = {}
        total_norm = 0.0
        for name, deltas in layer_deltas.items():
            # 该层在所有客户端上的平均范数
            mean_norm = torch.stack([d.norm(2) for d in deltas]).mean().item()
            log_dict[f'gradient/norm_l2/mean_{name}'] = mean_norm
            total_norm += mean_norm
        
        log_dict['gradient/total_norm_l2'] = total_norm
        
        # 裁剪比率估计
        if total_norm > 0:
            typical_max = 10.0
            clip_ratio = min(total_norm / typical_max, 1.0)
            log_dict['gradient/clipping_ratio'] = clip_ratio
        
        wandb.log(log_dict, step=step)
        
        # ── 客户端间梯度统计（需要 >= 2 个客户端）──
        if len(client_grads) > 1:
            self._log_client_gradient_stats(client_grads, step)
    
    def _log_client_gradient_stats(self, client_grads, step):
        """计算客户端间 weight delta 的方差和余弦相似度"""
        flat_grads = []
        for cg in client_grads:
            flat = []
            for name, delta in cg.items():
                if delta is not None:
                    flat.append(delta.reshape(-1).float())
            if flat:
                flat_grads.append(torch.cat(flat))
        
        if len(flat_grads) < 2:
            return
        
        stacked = torch.stack(flat_grads)  # [K, D]
        
        # 梯度方差（逐元素方差的均值）
        grad_variance = stacked.var(dim=0).mean().item()
        
        # 梯度信噪比 (GSNR): 均值² / 方差
        grad_mean_sq = stacked.mean(dim=0).norm(2).item() ** 2 / stacked.shape[1]
        gsnr = grad_mean_sq / grad_variance if grad_variance > 1e-10 else 0.0
        
        # 两两余弦相似度
        cos_sims = []
        for i in range(len(flat_grads)):
            for j in range(i + 1, len(flat_grads)):
                sim = F.cosine_similarity(
                    flat_grads[i].unsqueeze(0), flat_grads[j].unsqueeze(0)
                ).item()
                cos_sims.append(sim)
        
        log_dict = {
            'gradient/variance_mean': grad_variance,
            'gradient/gsnr': gsnr,
        }
        if cos_sims:
            log_dict['gradient/cos_sim_mean'] = float(np.mean(cos_sims))
            log_dict['gradient/cos_sim_min'] = float(np.min(cos_sims))
            wandb.log({'gradient/cos_sim_distribution': wandb.Histogram(np.array(cos_sims))}, step=step)
        
        wandb.log(log_dict, step=step)
    
    # ──────────────────────────────────
    # V3: 样本难度监控
    # ──────────────────────────────────
    def log_sample_metrics(self, step=None, per_sample_losses=None, per_sample_confs=None,
                           subgroup_labels=None, sensitive_labels=None):
        """
        记录样本级别的深度指标
        
        Args:
            step: 步数
            per_sample_losses: ndarray or tensor shape [N], 每个样本的 loss
            per_sample_confs: ndarray or tensor shape [N], 每个样本的预测置信度 (max softmax)
            subgroup_labels: ndarray or tensor shape [N], 样本所属的敏感属性子组标签 (0或1)
            sensitive_labels: 如果 subgroup_labels 未提供，可直接用此字段
        """
        if step is None:
            step = self.step
        
        log_dict = {}
        
        if per_sample_losses is not None:
            losses = np.asarray(per_sample_losses).flatten()
            losses = losses[~np.isnan(losses) & ~np.isinf(losses)]
            log_dict['sample/loss_mean'] = float(np.mean(losses))
            log_dict['sample/loss_std'] = float(np.std(losses))
            wandb.log({'sample/loss_distribution': wandb.Histogram(losses)}, step=step)
            
            # 记录 top-10% 难样本的 loss（高 loss 尾部）
            if len(losses) > 10:
                cutoff = np.percentile(losses, 90)
                hard_samples = losses[losses >= cutoff]
                log_dict['sample/loss_hard_top10_pct'] = float(np.mean(hard_samples))
        
        if per_sample_confs is not None:
            confs = np.asarray(per_sample_confs).flatten()
            confs = confs[~np.isnan(confs) & ~np.isinf(confs)]
            log_dict['sample/confidence_mean'] = float(np.mean(confs))
            wandb.log({'sample/confidence_distribution': wandb.Histogram(confs)}, step=step)
            
            # 低置信度样本比例
            low_conf_ratio = float(np.mean(confs < 0.6))
            log_dict['sample/low_confidence_ratio'] = low_conf_ratio
        
        # 按敏感属性分组统计 loss
        labels = sensitive_labels if sensitive_labels is not None else subgroup_labels
        if labels is not None and per_sample_losses is not None:
            labels = np.asarray(labels).flatten()
            losses = np.asarray(per_sample_losses).flatten()
            unique_groups = np.unique(labels)
            for g in unique_groups:
                mask = (labels == g)
                if mask.sum() > 0:
                    group_loss = float(np.mean(losses[mask]))
                    log_dict[f'sample/subgroup_loss/group_{int(g)}'] = group_loss
        
        if log_dict:
            wandb.log(log_dict, step=step)
    
    # ──────────────────────────────────
    # V3: 表征（Embedding）质量监控
    # ──────────────────────────────────
    def log_embedding_metrics(self, step=None, embeddings=None, labels=None,
                              sensitive_labels=None):
        """
        记录表征空间的深度指标
        
        Args:
            step: 步数
            embeddings: ndarray or tensor shape [N, D], 模型输出的embedding向量
            labels: ndarray or tensor shape [N], 分类标签
            sensitive_labels: ndarray or tensor shape [N], 敏感属性标签
        """
        if step is None:
            step = self.step
        
        if embeddings is None:
            return
        
        emb = np.asarray(embeddings)
        if emb.ndim != 2 or emb.shape[0] < 10:
            return
        
        log_dict = {}
        
        # 类内/类间距离比（按分类标签）
        if labels is not None:
            labs = np.asarray(labels).flatten()
            unique_labs = np.unique(labs)
            intra_dists, inter_dists = [], []
            
            for lab in unique_labs:
                mask = (labs == lab)
                if mask.sum() >= 2:
                    points = emb[mask]
                    # 类内：同类的 pairwise 距离
                    intra = torch.pdist(torch.tensor(points), p=2)
                    intra_dists.append(float(intra.mean()))
            
            # 类间：不同类质心之间的距离
            centroids = []
            for lab in unique_labs:
                mask = (labs == lab)
                if mask.sum() > 0:
                    centroids.append(emb[mask].mean(axis=0))
            if len(centroids) >= 2:
                centroids = torch.tensor(np.stack(centroids))
                inter = torch.pdist(centroids, p=2)
                inter_dists.append(float(inter.mean()))
            
            if intra_dists and inter_dists:
                intra_mean = np.mean(intra_dists)
                inter_mean = np.mean(inter_dists)
                ratio = inter_mean / (intra_mean + 1e-8)
                log_dict['embedding/intra_inter_distance_ratio'] = ratio
                log_dict['embedding/intra_class_distance'] = intra_mean
                log_dict['embedding/inter_class_distance'] = inter_mean
        
        # 敏感属性可分离度：在embedding上训一个简单线性分类器
        if sensitive_labels is not None:
            sens = np.asarray(sensitive_labels).flatten()
            unique_sens = np.unique(sens)
            if len(unique_sens) == 2 and emb.shape[0] > 10:
                try:
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(max_iter=200, C=1.0)
                    clf.fit(emb, sens)
                    score = clf.score(emb, sens)
                    log_dict['embedding/sensitive_separability'] = float(score)
                except Exception:
                    pass
        
        if log_dict:
            wandb.log(log_dict, step=step)
    
    # ──────────────────────────────────
    # V3: GPU 内存监控
    # ──────────────────────────────────
    def log_memory_metrics(self, step=None):
        """记录 GPU 内存使用情况"""
        if step is None:
            step = self.step
        if torch.cuda.is_available():
            log_dict = {}
            for i in range(torch.cuda.device_count()):
                try:
                    mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    log_dict[f'memory/gpu_{i}_allocated_gb'] = mem_alloc
                    log_dict[f'memory/gpu_{i}_reserved_gb'] = mem_reserved
                except Exception:
                    pass
            if log_dict:
                wandb.log(log_dict, step=step)
    
    # ────────────────────────────────────────────────────────────
    # V4: 参数重要性估计（Fisher对角线）— 启发自 EWC / Mang Ye 参数估计
    #   empirical Fisher: F_i ≈ E[(∂log p(y|x;θ)/∂θ_i)²]
    #   高 Fisher 的参数 = 对任务"重要"，FL 中可据此做稀疏通信
    # ────────────────────────────────────────────────────────────
    def log_fisher_diagonal(self, model, dataloader, param_dict, step=None,
                            max_samples=200, device=None):
        """估计模型参数的 empirical Fisher 对角线"""
        if step is None:
            step = self.step
        
        fisher_diag = {}
        try:
            model.eval()
            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            n_samples = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fisher_diag[name] = torch.zeros_like(param.data)
            
            with torch.no_grad():
                for batch in dataloader:
                    if n_samples >= max_samples:
                        break
                    
                    if device is None:
                        device = next(model.parameters()).device
                    
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    elif 'img' in batch:
                        imgs = batch['img'].to(device)
                        labels = batch['labels'].to(device)
                        if hasattr(model, 'forward'):
                            result = model(imgs)
                            if isinstance(result, tuple):
                                logits = result[0]
                            else:
                                logits = result
                        else:
                            logits = model(imgs)
                    elif 'X' in batch:
                        X = batch['X'].to(device)
                        labels = batch['labels'].to(device)
                        logits = model(X)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                    else:
                        continue
                    
                    # 对每个样本单独算梯度以得到逐参数的平方
                    log_probs = F.log_softmax(logits, dim=1)
                    for i in range(min(len(labels), max_samples - n_samples)):
                        model.zero_grad()
                        log_probs[i, labels[i]].backward(retain_graph=True)
                        for name, param in model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                fisher_diag[name] += param.grad.data ** 2
                        n_samples += 1
                    
                    model.zero_grad()
            
            # 聚合到wandb
            if n_samples > 0:
                log_dict = {}
                total = 0.0
                layer_importances = {}
                for name, f_val in fisher_diag.items():
                    f_mean = f_val.mean().item() / n_samples
                    layer_importances[name] = f_mean
                    total += f_val.sum().item()
                    log_dict[f'fisher/diag_mean/{name}'] = f_mean
                
                # 层级别的相对重要性
                if total > 0:
                    for name in layer_importances:
                        rel_imp = (fisher_diag[name].sum().item() / total) * 100
                        log_dict[f'fisher/relative_importance_pct/{name}'] = rel_imp
                
                # 全体 Fisher 信息量的分布（直方图）
                all_fisher = torch.cat([f.flatten() for f in fisher_diag.values()]).cpu().numpy() / n_samples
                wandb.log({'fisher/all_params_distribution': wandb.Histogram(all_fisher)}, step=step)
                log_dict['fisher/total_information'] = total / n_samples
                wandb.log(log_dict, step=step)
        except Exception:
            pass
        finally:
            model.zero_grad()
    
    # ────────────────────────────────────────────────────────────
    # V4: Loss Landscape Sharpness — 启发自 SAM (Foret et al. 2021)
    #   sharpness = max_{|ε|≤ρ} L(w+ε) - L(w)
    #   越平坦 = 泛化越好。FL 聚合后 sharpness 骤升 = 聚合有害
    # ────────────────────────────────────────────────────────────
    def log_loss_landscape_sharpness(self, model, dataloader, param_dict, step=None,
                                     rho=0.05, max_samples=100, device=None):
        """估计 loss landscape 的 sharpness"""
        if step is None:
            step = self.step
        try:
            if device is None:
                device = next(model.parameters()).device
            
            loss_fn = torch.nn.CrossEntropyLoss()
            model.eval()
            
            # 1) 记录原始权重和原始 loss
            original_weights = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    original_weights[name] = param.data.clone()
            
            orig_losses = []
            n_samples = 0
            with torch.no_grad():
                for batch in dataloader:
                    if n_samples >= max_samples:
                        break
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    elif 'img' in batch:
                        imgs = batch['img'].to(device)
                        labels = batch['labels'].to(device)
                        result = model(imgs)
                        logits = result[0] if isinstance(result, tuple) else result
                    elif 'X' in batch:
                        X = batch['X'].to(device)
                        labels = batch['labels'].to(device)
                        result = model(X)
                        logits = result[0] if isinstance(result, tuple) else result
                    else:
                        continue
                    loss = loss_fn(logits, labels.long()).item()
                    orig_losses.append(loss)
                    n_samples += len(labels)
            
            orig_loss = np.mean(orig_losses) if orig_losses else 0.0
            
            # 2) 计算扰动方向（梯度方向，用第一批数据近似）
            ascent_losses = []
            for batch in dataloader:
                if len(ascent_losses) >= 1:
                    break
                if 'input_ids' in batch:
                    input_ids = batch['input_ids'][:min(len(batch['input_ids']), 32)].to(device)
                    attention_mask = batch['attention_mask'][:min(len(batch['input_ids']), 32)].to(device)
                    labels = batch['labels'][:min(len(batch['input_ids']), 32)].to(device)
                    _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                elif 'img' in batch:
                    imgs = batch['img'][:32].to(device)
                    labels = batch['labels'][:32].to(device)
                    result = model(imgs)
                    logits = result[0] if isinstance(result, tuple) else result
                elif 'X' in batch:
                    X = batch['X'][:32].to(device)
                    labels = batch['labels'][:32].to(device)
                    result = model(X)
                    logits = result[0] if isinstance(result, tuple) else result
                else:
                    continue
                
                loss = loss_fn(logits, labels.long())
                grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
                
                # 3) 沿梯度方向扰动 rho 步长
                with torch.no_grad():
                    # 缩放梯度使其 L2 norm = rho
                    grad_norm = sum(g.norm(2) ** 2 for g in grads).sqrt()
                    if grad_norm > 0:
                        scale = rho / grad_norm
                        idx = 0
                        for name, param in model.named_parameters():
                            if param.requires_grad and idx < len(grads):
                                param.data += scale * grads[idx]
                                idx += 1
                
                # 4) 计算扰动后的 loss
                with torch.no_grad():
                    for batch2 in dataloader:
                        if len(ascent_losses) >= max_samples:
                            break
                        # (batch processing same as above, simplified)
                        pass
                
                break
            
            # 简化版：直接算 sharpness = max(loss_diff) on batch
            sharpness_list = []
            with torch.no_grad():
                for batch in dataloader:
                    if n_samples >= max_samples:
                        break
                    # compute loss with perturbed weights
                    if 'input_ids' in batch:
                        _, logits = model(input_ids=batch['input_ids'].to(device)[:32],
                                         attention_mask=batch['attention_mask'].to(device)[:32])
                        labels = batch['labels'].to(device)[:32]
                    elif 'img' in batch:
                        result = model(batch['img'].to(device)[:32])
                        logits = result[0] if isinstance(result, tuple) else result
                        labels = batch['labels'].to(device)[:32]
                    elif 'X' in batch:
                        result = model(batch['X'].to(device)[:32])
                        logits = result[0] if isinstance(result, tuple) else result
                        labels = batch['labels'].to(device)[:32]
                    else:
                        continue
                    loss_perturbed = loss_fn(logits, labels.long()).item()
                    sharpness_list.append(loss_perturbed - orig_loss)
                    n_samples += 32
            
            # 5) 恢复原始权重
            for name, param in model.named_parameters():
                if name in original_weights:
                    param.data = original_weights[name]
            
            model.zero_grad()
            
            if sharpness_list:
                sharpness = np.mean(sharpness_list)
                wandb.log({
                    'landscape/sharpness': sharpness,
                    'landscape/sharpness_normalized': sharpness / (orig_loss + 1e-8),
                    'landscape/original_loss': orig_loss
                }, step=step)
        except Exception:
            model.zero_grad()
            try:
                for name, param in model.named_parameters():
                    if name in original_weights:
                        param.data = original_weights[name]
            except Exception:
                pass
    
    # ────────────────────────────────────────────────────────────
    # V4: 层级别更新稀疏度 + 参数稳定性指数
    #   启发自 FL compression 文献 + Mang Ye 参数估计
    # ────────────────────────────────────────────────────────────
    def log_update_statistics(self, model, previous_weights=None, step=None,
                              sparsity_threshold=1e-5):
        """记录模型更新的稀疏度和参数稳定性"""
        if step is None:
            step = self.step
        
        if previous_weights is None:
            return
        
        try:
            total_params = 0
            near_zero_updates = 0
            log_dict = {}
            
            for name, param in model.named_parameters():
                if param.requires_grad and name in previous_weights:
                    update = param.data - previous_weights[name]
                    abs_update = update.abs()
                    n_params = param.numel()
                    total_params += n_params
                    
                    # 稀疏度：接近零的更新比例
                    n_sparse = (abs_update < sparsity_threshold).sum().item()
                    near_zero_updates += n_sparse
                    
                    sparsity_ratio = n_sparse / n_params if n_params > 0 else 0
                    log_dict[f'update/sparsity_ratio/{name}'] = sparsity_ratio
                    
                    # 更新的统计量
                    update_mean = abs_update.mean().item()
                    log_dict[f'update/magnitude_mean/{name}'] = update_mean
            
            overall_sparsity = near_zero_updates / total_params if total_params > 0 else 0
            log_dict['update/overall_sparsity_ratio'] = overall_sparsity
            
            # 参数稳定性指数：更新幅度 / 参数幅度
            for name, param in model.named_parameters():
                if param.requires_grad and name in previous_weights:
                    param_magnitude = param.data.abs().mean().item()
                    update_magnitude = (param.data - previous_weights[name]).abs().mean().item()
                    if param_magnitude > 1e-10:
                        stability = 1.0 - (update_magnitude / param_magnitude)
                        stability = max(0.0, min(1.0, stability))
                        log_dict[f'update/stability_index/{name}'] = stability
            
            # 更新熵（信息论角度）
            if total_params > 0 and near_zero_updates < total_params:
                all_updates = []
                for name, param in model.named_parameters():
                    if param.requires_grad and name in previous_weights:
                        all_updates.append((param.data - previous_weights[name]).abs().flatten())
                if all_updates:
                    all_u = torch.cat(all_updates)
                    all_u = all_u[all_u > 0]  # 避免 log(0)
                    if len(all_u) > 0:
                        probs = all_u / all_u.sum()
                        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                        max_entropy = np.log(len(all_u))
                        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        log_dict['update/entropy_normalized'] = normalized_entropy
            
            if log_dict:
                wandb.log(log_dict, step=step)
        except Exception:
            pass
    
    # ────────────────────────────────────────────────────────────
    # V4: 激活统计 — 死神经元 / 激活饱和度 / 激活熵
    #   启发自 Kurtz 2020, Liu 2023 的稀疏训练文献
    # ────────────────────────────────────────────────────────────
    def log_activation_statistics(self, model, dataloader, param_dict, step=None,
                                  max_samples=200, device=None):
        """记录激活值的统计信息（需通过 hook 收集）"""
        if step is None:
            step = self.step
        try:
            if device is None:
                device = next(model.parameters()).device
            model.eval()
            
            # 为 ReLU / GELU 层注册 hook 收集激活值
            activations = {}
            hooks = []
            
            def make_hook(name):
                def hook_fn(module, input, output):
                    activations.setdefault(name, []).append(output.detach().cpu())
                return hook_fn
            
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
                    hooks.append(module.register_forward_hook(make_hook(name)))
            
            # 跑一个 batch 收集激活
            n_samples = 0
            with torch.no_grad():
                for batch in dataloader:
                    if n_samples >= max_samples:
                        break
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'][:min(len(batch['input_ids']), 32)].to(device)
                        attention_mask = batch['attention_mask'][:min(len(batch['attention_mask']), 32)].to(device)
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    elif 'img' in batch:
                        model(batch['img'][:32].to(device))
                    elif 'X' in batch:
                        model(batch['X'][:32].to(device))
                    n_samples += 32
            
            # 分析激活
            log_dict = {}
            for name, act_list in activations.items():
                if act_list:
                    act = torch.cat([a.flatten() for a in act_list])
                    # 死神经元比例（激活全零的神经元）
                    if act_list[0].ndim >= 2:
                        dead_ratio = (act_list[0].sum(dim=(0, 1)) == 0).float().mean().item()
                        log_dict[f'activation/dead_neuron_ratio/{name}'] = dead_ratio
                    
                    # 激活均值/方差
                    log_dict[f'activation/mean/{name}'] = act.mean().item()
                    log_dict[f'activation/std/{name}'] = act.std().item()
                    
                    # 激活熵（更高的熵 = 信息更分散）
                    act_pos = act[act > 0]
                    if len(act_pos) > 0:
                        probs = act_pos / act_pos.sum()
                        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                        max_entropy = np.log(len(act_pos))
                        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        log_dict[f'activation/entropy/{name}'] = norm_entropy
            
            if log_dict:
                wandb.log(log_dict, step=step)
            
            for h in hooks:
                h.remove()
        except Exception:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
    
    # ────────────────────────────────────────────────────────────
    # V4: Neural Collapse 指标 — 启发自 Papyan et al. 2020 (PNAS)
    #   训练收敛时表征会坍缩到类均值 (Simplex ETF)
    # ────────────────────────────────────────────────────────────
    def log_neural_collapse_metrics(self, embeddings, labels, step=None):
        """记录 Neural Collapse 相关指标"""
        if step is None:
            step = self.step
        if embeddings is None or labels is None:
            return
        try:
            emb = np.asarray(embeddings)
            labs = np.asarray(labels).flatten()
            unique_labs = np.unique(labs)
            if len(unique_labs) < 2:
                return
            
            # 1) 各类的类内方差
            within_vars = []
            class_means = []
            for lab in unique_labs:
                mask = (labs == lab)
                if mask.sum() >= 2:
                    points = emb[mask]
                    class_mean = points.mean(axis=0)
                    class_means.append(class_mean)
                    within_var = np.mean(np.sum((points - class_mean) ** 2, axis=1))
                    within_vars.append(within_var)
            
            log_dict = {}
            if within_vars:
                log_dict['neural_collapse/within_class_variance'] = np.mean(within_vars)
            
            # 2) 各类均值是否等距且等范数（ETF 结构的检测）
            if len(class_means) >= 3:
                class_means = np.stack(class_means)  # [C, D]
                # 归一化类均值
                norms = np.linalg.norm(class_means, axis=1)
                norm_std = np.std(norms) / (np.mean(norms) + 1e-8)
                log_dict['neural_collapse/centroid_norm_equality'] = norm_std
                
                # 类均值之间夹角的均匀性
                class_means_norm = class_means / (np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8)
                gram = class_means_norm @ class_means_norm.T  # [C, C]
                off_diag = gram[~np.eye(len(unique_labs), dtype=bool)]
                if len(off_diag) > 0:
                    max_sim = np.max(off_diag)
                    log_dict['neural_collapse/max_inter_class_cosine'] = float(max_sim)
                    
                    # ETF: 理想情况下所有类间余弦应相等 = -1/(C-1)
                    C = len(unique_labs)
                    ideal_cos = -1 / (C - 1)
                    cos_deviation = np.std(off_diag - ideal_cos)
                    log_dict['neural_collapse/etf_deviation'] = float(cos_deviation)
            
            if log_dict:
                wandb.log(log_dict, step=step)
        except Exception:
            pass
    
    # ────────────────────────────────────────────────────────────
    # V4: 客户端数据分布差异估计
    #   用 label 分布和敏感属性分布的 JS 散度近似
    # ────────────────────────────────────────────────────────────
    def log_client_distribution_divergence(self, step=None, client_label_counts=None,
                                           client_sa_counts=None, num_classes=2):
        """估计客户端间数据分布的差异"""
        if step is None:
            step = self.step
        if client_label_counts is None:
            return
        try:
            # client_label_counts: list of arrays, each [num_classes]
            K = len(client_label_counts)
            if K < 2:
                return
            
            # 全局平均分布
            global_dist = np.mean(np.stack(client_label_counts), axis=0)
            global_dist = global_dist / (global_dist.sum() + 1e-8)
            
            # 每个客户端与全局分布的 JS 散度
            js_divs = []
            for i in range(K):
                client_dist = np.asarray(client_label_counts[i], dtype=float)
                client_dist = client_dist / (client_dist.sum() + 1e-8)
                m = (global_dist + client_dist) / 2
                kl_c = np.sum(client_dist * np.log((client_dist + 1e-8) / (m + 1e-8)))
                kl_g = np.sum(global_dist * np.log((global_dist + 1e-8) / (m + 1e-8)))
                js = (kl_c + kl_g) / 2
                js_divs.append(js)
            
            log_dict = {
                'client/distribution_js_div_mean': float(np.mean(js_divs)),
                'client/distribution_js_div_max': float(np.max(js_divs)),
            }
            wandb.log({'client/distribution_js_div_all': wandb.Histogram(np.array(js_divs))}, step=step)
            
            # 敏感属性分布差异
            if client_sa_counts is not None:
                sa_js_divs = []
                global_sa = np.mean(np.stack(client_sa_counts), axis=0)
                global_sa = global_sa / (global_sa.sum() + 1e-8)
                for i in range(K):
                    sa_dist = np.asarray(client_sa_counts[i], dtype=float)
                    sa_dist = sa_dist / (sa_dist.sum() + 1e-8)
                    m = (global_sa + sa_dist) / 2
                    kl_c = np.sum(sa_dist * np.log((sa_dist + 1e-8) / (m + 1e-8)))
                    kl_g = np.sum(global_sa * np.log((global_sa + 1e-8) / (m + 1e-8)))
                    sa_js_divs.append((kl_c + kl_g) / 2)
                
                log_dict['client/sa_distribution_js_div_mean'] = float(np.mean(sa_js_divs))
            
            if log_dict:
                wandb.log(log_dict, step=step)
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────
    # 监控配置（全部默认开启，可通过 param_dict['tb_monitor'] 覆盖）
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def get_monitoring_config(param_dict):
        """全部默认开启。如需关闭某项，在 param_dict['tb_monitor'] 中设置。"""
        default = {
            'test':               True,   'test_freq':               1,
            'system':             True,   'system_freq':             1,
            'gradient':           True,   'gradient_freq':           1,
            'embedding':          True,   'embedding_freq':          1,   'embedding_samples':   300,
            'neural_collapse':    True,   'neural_collapse_freq':    1,
            'fisher':             True,   'fisher_freq':             1,   'fisher_samples':       100,
            'sharpness':          True,   'sharpness_freq':          1,   'sharpness_samples':     64,
            'activation':         True,   'activation_freq':         1,   'activation_samples':   100,
            'update_stats':       True,   'update_stats_freq':       1,
            'client_divergence':  True,   'client_divergence_freq':  1,
            'deep_log_freq':      1,
        }
        user_config = param_dict.get('tb_monitor', {})
        default.update(user_config)
        return default

    # ──────────────────────────────────
    # 超参数记录
    # ──────────────────────────────────
    def log_hyperparameters(self, hparam_dict, metric_dict):
        """wandb在init时已经记录了config，这里可以额外更新"""
        pass
    
    def log_experiment_config(self, param_dict):
        """以文本形式记录实验配置（wandb会自动记录config）"""
        pass

    # ──────────────────────────────────
    # 工具方法
    # ──────────────────────────────────
    def update_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def add_text(self, tag, text_string, step=None):
        """wandb不支持直接添加文本，可以用wandb.Table或markdown"""
        pass
    
    def flush(self):
        """wandb自动flush"""
        pass
    
    def close(self):
        """关闭wandb run"""
        if self.run is not None:
            self.run.finish()
            self.run = None


# ────────────────────────────────────────────────────
# 全局实例管理
# ────────────────────────────────────────────────────
_wandb_logger = None


def init_wandb_logger(project=None, experiment_name=None, algorithm=None, dataset=None,
                      config=None, mode='online'):
    global _wandb_logger
    _wandb_logger = WandbLogger(
        project=project,
        experiment_name=experiment_name,
        algorithm=algorithm,
        dataset=dataset,
        config=config,
        mode=mode
    )
    return _wandb_logger


def get_wandb_logger():
    global _wandb_logger
    return _wandb_logger


def log_scalar(tag, value, step=None):
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_scalar(tag, value, step)


def log_metrics(metrics_dict, step=None, prefix=''):
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_metrics(metrics_dict, step, prefix)


def log_test_metrics(accuracy=None, DEO=None, SPD=None, FR=None, HM=None,
                     step=None, gpu_seconds=None, avg_gpu_seconds=None,
                     communication_cost=None, prefix='test/'):
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_test_metrics(
            accuracy=accuracy, DEO=DEO, SPD=SPD, FR=FR, HM=HM,
            step=step, gpu_seconds=gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
            communication_cost=communication_cost, prefix=prefix
        )


def log_system_metrics(step=None, gpu_seconds=None, communication_cost=None,
                       selected_client_count=None, model_mb_size=None):
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_system_metrics(
            step=step, gpu_seconds=gpu_seconds, communication_cost=communication_cost,
            selected_client_count=selected_client_count, model_mb_size=model_mb_size
        )


def log_client_metrics(step=None, **kwargs):
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_client_metrics(step=step, **kwargs)


def log_model_weights(model, step=None):
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_model_weights(model, step)


def log_gradient_metrics(model, step=None, client_grads=None):
    """V3: 记录梯度深度指标（基于客户端 weight delta）"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_gradient_metrics(model, step=step, client_grads=client_grads)


def log_sample_metrics(step=None, per_sample_losses=None, per_sample_confs=None,
                       subgroup_labels=None, sensitive_labels=None):
    """V3: 记录样本难度深度指标"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_sample_metrics(
            step=step, per_sample_losses=per_sample_losses,
            per_sample_confs=per_sample_confs,
            subgroup_labels=subgroup_labels, sensitive_labels=sensitive_labels
        )


def log_embedding_metrics(step=None, embeddings=None, labels=None, sensitive_labels=None):
    """V3: 记录表征质量深度指标"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_embedding_metrics(
            step=step, embeddings=embeddings,
            labels=labels, sensitive_labels=sensitive_labels
        )


def log_memory_metrics(step=None):
    """V3: 记录GPU内存使用"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_memory_metrics(step=step)


def log_fisher_diagonal(model, dataloader, param_dict, step=None, max_samples=200, device=None):
    """V4: 估计 Fisher 对角线（参数重要性）— EWC / Mang Ye 参数估计"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_fisher_diagonal(model, dataloader, param_dict, step=step, max_samples=max_samples, device=device)


def log_loss_landscape_sharpness(model, dataloader, param_dict, step=None, rho=0.05, max_samples=100, device=None):
    """V4: Loss landscape sharpness — SAM (Foret et al. 2021)"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_loss_landscape_sharpness(model, dataloader, param_dict, step=step, rho=rho, max_samples=max_samples, device=device)


def log_update_statistics(model, previous_weights=None, step=None, sparsity_threshold=1e-5):
    """V4: 模型更新稀疏度 + 参数稳定性指数 — FL compression / Mang Ye"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_update_statistics(model, previous_weights=previous_weights, step=step, sparsity_threshold=sparsity_threshold)


def log_activation_statistics(model, dataloader, param_dict, step=None, max_samples=200, device=None):
    """V4: 激活统计（死神经元/饱和度/熵）— Kurtz 2020, Liu 2023"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_activation_statistics(model, dataloader, param_dict, step=step, max_samples=max_samples, device=device)


def log_neural_collapse_metrics(embeddings, labels, step=None):
    """V4: Neural Collapse 指标 — Papyan et al. 2020 (PNAS)"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_neural_collapse_metrics(embeddings, labels, step=step)


def log_client_distribution_divergence(step=None, client_label_counts=None, client_sa_counts=None, num_classes=2):
    """V4: 客户端数据分布差异估计"""
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.log_client_distribution_divergence(step=step, client_label_counts=client_label_counts, client_sa_counts=client_sa_counts, num_classes=num_classes)


def get_monitoring_config(param_dict):
    """获取监控配置（全部默认开启，可通过 param_dict['tb_monitor'] 关闭某项）"""
    return WandbLogger.get_monitoring_config(param_dict)


def log_deep_metrics(global_model, param_dict, testing_dataloader, step,
                     client_model_updates=None):
    """
    深度监控（V3+V4）：根据 tb_monitor 配置选择性开启各模块。
    调用方需确保外层已做 deep_log_freq 频率控制。
    所有计算复用 testing_dataloader。
    
    Args:
        global_model: 聚合后的全局模型
        param_dict: 实验参数字典
        testing_dataloader: 测试数据加载器
        step: 当前通信轮次
        client_model_updates: 可选，各客户端本地训练后的权重变化列表
            [{name: tensor_delta, ...}, ...]，用于客户端间梯度方差/余弦相似度
    """
    import torch
    import numpy as np
    
    cfg = get_monitoring_config(param_dict)
    device = param_dict['device']
    
    if cfg.get('gradient') and step % cfg.get('gradient_freq', 5) == 0:
        log_gradient_metrics(global_model, step=step, client_grads=client_model_updates)
    
    if cfg.get('system') and step % cfg.get('system_freq', 1) == 0:
        log_memory_metrics(step=step)
    
    if cfg.get('fisher') and step % cfg.get('fisher_freq', 10) == 0:
        log_fisher_diagonal(global_model, testing_dataloader, param_dict,
                           step=step, max_samples=cfg.get('fisher_samples', 100), device=device)
    
    if cfg.get('sharpness') and step % cfg.get('sharpness_freq', 15) == 0:
        try:
            log_loss_landscape_sharpness(global_model, testing_dataloader, param_dict,
                                        step=step, rho=0.05,
                                        max_samples=cfg.get('sharpness_samples', 64), device=device)
        except Exception:
            pass
    
    need_embedding = cfg.get('embedding') and step % cfg.get('embedding_freq', 5) == 0
    need_neural = cfg.get('neural_collapse') and step % cfg.get('neural_collapse_freq', 5) == 0
    need_activation = cfg.get('activation') and step % cfg.get('activation_freq', 10) == 0
    
    if need_embedding or need_neural:
        try:
            global_model.eval()
            embeddings_list, labels_list, sa_list = [], [], []
            max_samples = cfg.get('embedding_samples', 300)
            
            with torch.no_grad():
                for d in testing_dataloader:
                    if len(labels_list) >= max_samples:
                        break
                    # 支持多种 dataloader 格式
                    if 'input_ids' in d:
                        input_ids = d["input_ids"].to(device)
                        attention_mask = d["attention_mask"].to(device)
                        labels = d["labels"]
                        sa = d["protected"]
                        outputs = global_model.bert(input_ids=input_ids, attention_mask=attention_mask)
                    elif 'img' in d:
                        result = global_model(d["img"].to(device))
                        labels = d["labels"]
                        sa = d["protected"]
                        if isinstance(result, tuple):
                            outputs = result[0]
                        else:
                            outputs = result
                        # 对于非BERT模型（如 ResNet），直接用 logits 前的特征
                        emb = outputs if isinstance(outputs, torch.Tensor) else None
                        if emb is not None:
                            embeddings_list.append(emb.cpu().detach())
                            labels_list.extend(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
                            sa_list.extend(sa.tolist() if hasattr(sa, 'tolist') else list(sa))
                        continue
                    elif 'X' in d:
                        result = global_model(d["X"].to(device))
                        labels = d["labels"]
                        sa = d["protected"]
                        if isinstance(result, tuple):
                            emb = result[0].cpu().detach() if hasattr(result[0], 'cpu') else result[0]
                        else:
                            emb = result.cpu().detach() if hasattr(result, 'cpu') else result
                        embeddings_list.append(emb)
                        labels_list.extend(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
                        sa_list.extend(sa.tolist() if hasattr(sa, 'tolist') else list(sa))
                        continue
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        emb = outputs.last_hidden_state.mean(dim=1).cpu()
                    else:
                        emb = outputs[0].mean(dim=1).cpu()
                    
                    embeddings_list.append(emb)
                    labels_list.extend(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
                    sa_list.extend(sa.tolist() if hasattr(sa, 'tolist') else list(sa))
            
            if embeddings_list:
                all_emb = torch.cat(embeddings_list, dim=0).numpy()
                labs_arr = np.array(labels_list[:len(all_emb)])
                sa_arr = np.array(sa_list[:len(all_emb)])
                
                if need_embedding:
                    log_embedding_metrics(step=step, embeddings=all_emb,
                                         labels=labs_arr, sensitive_labels=sa_arr)
                if need_neural:
                    log_neural_collapse_metrics(embeddings=all_emb, labels=labs_arr)
            global_model.train()
        except Exception:
            pass
    
    if need_activation:
        try:
            log_activation_statistics(global_model, testing_dataloader, param_dict,
                                     step=step, max_samples=cfg.get('activation_samples', 100), device=device)
        except Exception:
            pass
    
    if cfg.get('update_stats') and step % cfg.get('update_stats_freq', 5) == 0:
        if not hasattr(log_deep_metrics, '_prev_weights'):
            log_deep_metrics._prev_weights = {}
        if log_deep_metrics._prev_weights and step > cfg.get('update_stats_freq', 5):
            log_update_statistics(global_model,
                                 previous_weights=log_deep_metrics._prev_weights, step=step)
        log_deep_metrics._prev_weights = {
            name: param.data.clone()
            for name, param in global_model.named_parameters() if param.requires_grad
        }


def update_step(step=None):
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.update_step(step)


def flush():
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.flush()


def close():
    global _wandb_logger
    if _wandb_logger is not None:
        _wandb_logger.close()
        _wandb_logger = None
