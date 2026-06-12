# https://arxiv.org/pdf/2503.17231
# LoGoFair: Post-Processing for Local and Global Fairness in Federated Learning

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


def get_model_predictions(param_dict, device, model, dataloader):
    """
    获取模型在数据集上的预测结果和敏感属性
    返回: predictions, labels, sensitive_attributes
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_sensitive = []
    
    with torch.no_grad():
        for batch in dataloader:
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.softmax(dim=1)[:, 1]
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
                tmp_logits, _, _ = model(imgs, return_logit=True)
                tmp_logits_min = tmp_logits.min()
                tmp_logits_max = tmp_logits.max()
                tmp_logits_range = tmp_logits_max - tmp_logits_min
                if tmp_logits_range > 0:
                    preds = (tmp_logits - tmp_logits_min) / tmp_logits_range
                else:
                    preds = torch.zeros_like(tmp_logits)
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)
                if "ANN" in str(type(model)):
                    local_prediction, _ = model(X)
                elif "LogisticRegression" in str(type(model)):
                    local_prediction = model(X)
                else:
                    local_prediction = model(X)
                preds = local_prediction.squeeze(1)
            
            labels = batch["labels"]
            sensitive = batch["protected"]
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_sensitive.append(sensitive.cpu())
    
    return torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_sensitive)


def compute_fairness_metrics(preds, labels, sensitive, threshold=0.5):
    """
    计算公平性指标 (DP 和 EO)
    """
    binary_preds = (preds >= threshold).float()
    
    # Demographic Parity
    pos_rate_group0 = (binary_preds[sensitive == 0].mean()).item() if (sensitive == 0).sum() > 0 else 0.0
    pos_rate_group1 = (binary_preds[sensitive == 1].mean()).item() if (sensitive == 1).sum() > 0 else 0.0
    dp = abs(pos_rate_group0 - pos_rate_group1)
    
    # Equalized Odds
    eo_values = []
    for y in [0, 1]:
        mask_y0 = (labels == y) & (sensitive == 0)
        mask_y1 = (labels == y) & (sensitive == 1)
        if mask_y0.sum() > 0 and mask_y1.sum() > 0:
            tpr_y0 = binary_preds[mask_y0].mean().item()
            tpr_y1 = binary_preds[mask_y1].mean().item()
            eo_values.append(abs(tpr_y0 - tpr_y1))
    
    eo = max(eo_values) if eo_values else 0.0
    
    return dp, eo


def calibrate_predictions_with_thresholds(preds, sensitive, thresholds):
    """
    使用群体特定阈值校准预测
    thresholds: dict {(group, label): threshold}
    """
    calibrated_preds = preds.clone()
    
    for g in [0, 1]:
        mask = (sensitive == g)
        if mask.sum() > 0:
            threshold = thresholds.get((g, 1), 0.5)
            calibrated_preds[mask] = (preds[mask] >= threshold).float()
    
    return calibrated_preds


def find_optimal_thresholds_for_local_fairness(preds, labels, sensitive, fairness_type="dp"):
    """
    为每个客户端寻找最优阈值以满足局部公平性
    使用网格搜索寻找最佳阈值组合
    """
    best_thresholds = {}
    
    for g in [0, 1]:
        mask_g = (sensitive == g)
        if mask_g.sum() == 0:
            best_thresholds[(g, 1)] = 0.5
            continue
        
        group_preds = preds[mask_g]
        group_labels = labels[mask_g]
        
        best_threshold = 0.5
        best_score = float('inf')
        
        # 网格搜索阈值
        for t in torch.arange(0.3, 0.7, 0.02):
            binary_preds = (group_preds >= t).float()
            
            if fairness_type == "dp":
                pos_rate = binary_preds.mean().item()
                # 目标：使正类预测率接近全局正类率
                global_pos_rate = (labels == 1).float().mean().item()
                score = abs(pos_rate - global_pos_rate)
            else:  # eo
                # 对于 EO，需要平衡 TPR 和 FPR
                mask_pos = (group_labels == 1)
                mask_neg = (group_labels == 0)
                
                if mask_pos.sum() > 0 and mask_neg.sum() > 0:
                    tpr = binary_preds[mask_pos].mean().item()
                    fpr = binary_preds[mask_neg].mean().item()
                    # 目标：使 TPR 和 FPR 的差异最小化
                    score = abs(tpr - fpr)
                else:
                    score = float('inf')
            
            if score < best_score:
                best_score = score
                best_threshold = t.item()
        
        best_thresholds[(g, 1)] = best_threshold
    
    return best_thresholds


def LoGoFair_post_processing(param_dict, device, global_model, training_dataloaders, num_clients_K):
    """
    LoGoFair 后处理阶段
    在 FL 训练完成后，对全局模型进行公平校准
    """
    logger.info("Starting LoGoFair post-processing...")
    
    # 步骤1: 收集全局模型的预测结果
    all_preds = []
    all_labels = []
    all_sensitive = []
    
    for client_id in range(num_clients_K):
        dataloader = training_dataloaders[client_id]
        preds, labels, sensitive = get_model_predictions(param_dict, device, global_model, dataloader)
        all_preds.append(preds)
        all_labels.append(labels)
        all_sensitive.append(sensitive)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_sensitive = torch.cat(all_sensitive)
    
    # 步骤2: 计算全局公平性指标
    global_dp, global_eo = compute_fairness_metrics(all_preds, all_labels, all_sensitive)
    logger.info(f"Before LoGoFair calibration - Global DP: {global_dp:.4f}, Global EO: {global_eo:.4f}")
    
    # 步骤3: 为每个客户端寻找最优局部阈值
    client_thresholds = []
    for client_id in range(num_clients_K):
        dataloader = training_dataloaders[client_id]
        preds, labels, sensitive = get_model_predictions(param_dict, device, global_model, dataloader)
        
        # 寻找最优阈值
        thresholds = find_optimal_thresholds_for_local_fairness(preds, labels, sensitive, fairness_type="dp")
        client_thresholds.append(thresholds)
    
    # 步骤4: 聚合全局阈值（用于全局公平性优化）
    global_thresholds = {}
    for g in [0, 1]:
        threshold_values = [ct[(g, 1)] for ct in client_thresholds]
        global_thresholds[(g, 1)] = np.mean(threshold_values)
    
    logger.info(f"LoGoFair global thresholds: {global_thresholds}")
    
    # 步骤5: 应用校准后的阈值进行预测
    calibrated_preds = calibrate_predictions_with_thresholds(all_preds, all_sensitive, global_thresholds)
    
    # 步骤6: 评估校准后的公平性
    calibrated_dp, calibrated_eo = compute_fairness_metrics(calibrated_preds, all_labels, all_sensitive)
    calibrated_accuracy = ((calibrated_preds == all_labels.float()).sum() / len(all_labels)).item()
    
    logger.info(f"After LoGoFair calibration - Accuracy: {calibrated_accuracy:.4f}, DP: {calibrated_dp:.4f}, EO: {calibrated_eo:.4f}")
    
    return global_thresholds, calibrated_accuracy, calibrated_dp, calibrated_eo


def LoGoFair(device,
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
    """
    LoGoFair: 后处理联邦公平学习方法
    
    核心思想：
    1. 先使用标准 FL（如 FedAvg）训练模型
    2. 训练完成后，对预训练模型进行后处理校准
    3. 通过群体特定阈值优化，同时实现局部和全局公平性
    
    参数：
    - post_processing_epochs: 后处理优化轮次
    - fairness_type: 公平性类型 ("dp" 或 "eo")
    - threshold_lr: 阈值学习率
    """
    
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    
    basic_path = param_dict['model_path']
    
    # LoGoFair 超参数
    post_processing_epochs = param_dict.get('loGoFair_post_epochs', 5)
    fairness_type = param_dict.get('loGoFair_fairness_type', 'dp')
    threshold_lr = param_dict.get('loGoFair_threshold_lr', 0.05)
    
    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):
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
    
    # 初始化阈值参数（用于后处理）
    global_thresholds = {0: 0.5, 1: 0.5}
    
    # 标准 FL 训练循环
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
        
        # Simulate Client Parallel
        for id in idxs_users:
            # Local Initialization
            logger.info("Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]
            
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
                    
                    if "SENT_CLF" in param_dict["task"]:
                        features, logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
                    loss.backward()
                    
                    if (batch_id + 1) % accumulation_steps == 0:
                        optimizer.step()
                        model.zero_grad()
                    
                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)
                    
                    epoch_total_loss += loss
                    
                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels
                    
                    gc.collect()
                
                if (batch_id + 1) % accumulation_steps != 0:
                    optimizer.step()
                    model.zero_grad()
                
                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")
            
            # Save local model
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            torch.save(model.cpu(), client_model_path)
            
            del model
            gc.collect()
        
        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")
        
        # Global operation - Parameter aggregation
        logger.info("Parameter aggregation")
        theta_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            theta_list.append(get_parameters(selected_model))
            del selected_model
            gc.collect()
        
        theta_list = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        
        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)
        
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")
        
        del theta_list
        gc.collect()
        
        # 测试
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time
            )
            clean_old_checkpoints(param_dict, keep_latest=5)
    
    # ========================================
    # LoGoFair 后处理阶段
    # ========================================
    logger.info("=" * 50)
    logger.info("Starting LoGoFair Post-Processing Phase")
    logger.info("=" * 50)
    
    global_model.eval()
    global_model.to(device)
    
    # 收集所有客户端的预测结果
    all_client_preds = []
    all_client_labels = []
    all_client_sensitive = []
    
    for client_id in range(num_clients_K):
        preds, labels, sensitive = get_model_predictions(param_dict, device, global_model, training_dataloaders[client_id])
        all_client_preds.append(preds)
        all_client_labels.append(labels)
        all_client_sensitive.append(sensitive)
    
    # 优化阈值参数
    threshold_params = torch.tensor([0.5, 0.5], requires_grad=True, device=device)
    threshold_optimizer = torch.optim.SGD([threshold_params], lr=threshold_lr)
    
    for pp_epoch in range(post_processing_epochs):
        threshold_optimizer.zero_grad()
        
        # 计算局部公平性损失
        local_fairness_loss = torch.tensor(0.0, device=device)
        local_accuracy_loss = torch.tensor(0.0, device=device)
        
        for client_id in range(num_clients_K):
            preds = all_client_preds[client_id].to(device)
            labels = all_client_labels[client_id].to(device)
            sensitive = all_client_sensitive[client_id].to(device)
            
            # 应用群体特定阈值
            calibrated_preds = preds.clone()
            for g in [0, 1]:
                mask = (sensitive == g)
                if mask.sum() > 0:
                    threshold = threshold_params[g]
                    # 使用 sigmoid 平滑近似阈值操作
                    calibrated_preds[mask] = torch.sigmoid((preds[mask] - threshold) * 10)
            
            # 局部公平性损失 (Demographic Parity)
            pos_rate_g0 = calibrated_preds[sensitive == 0].mean() if (sensitive == 0).sum() > 0 else torch.tensor(0.0)
            pos_rate_g1 = calibrated_preds[sensitive == 1].mean() if (sensitive == 1).sum() > 0 else torch.tensor(0.0)
            local_dp_loss = (pos_rate_g0 - pos_rate_g1) ** 2
            
            # 局部准确率损失
            local_acc_loss = torch.nn.BCELoss()(calibrated_preds.squeeze(), labels.float())
            
            local_fairness_loss += local_dp_loss
            local_accuracy_loss += local_acc_loss
        
        # 计算全局公平性损失
        all_preds_concat = torch.cat(all_client_preds).to(device)
        all_labels_concat = torch.cat(all_client_labels).to(device)
        all_sensitive_concat = torch.cat(all_client_sensitive).to(device)
        
        global_calibrated_preds = all_preds_concat.clone()
        for g in [0, 1]:
            mask = (all_sensitive_concat == g)
            if mask.sum() > 0:
                threshold = threshold_params[g]
                global_calibrated_preds[mask] = torch.sigmoid((all_preds_concat[mask] - threshold) * 10)
        
        global_pos_rate_g0 = global_calibrated_preds[all_sensitive_concat == 0].mean() if (all_sensitive_concat == 0).sum() > 0 else torch.tensor(0.0)
        global_pos_rate_g1 = global_calibrated_preds[all_sensitive_concat == 1].mean() if (all_sensitive_concat == 1).sum() > 0 else torch.tensor(0.0)
        global_dp_loss = (global_pos_rate_g0 - global_pos_rate_g1) ** 2
        
        global_acc_loss = torch.nn.BCELoss()(global_calibrated_preds.squeeze(), all_labels_concat.float())
        
        # 总损失：平衡公平性和准确率
        alpha = 0.5  # 公平性权重
        total_loss = alpha * (local_fairness_loss / num_clients_K + global_dp_loss) + (1 - alpha) * (local_accuracy_loss / num_clients_K + global_acc_loss)
        
        total_loss.backward()
        threshold_optimizer.step()
        
        # 约束阈值在合理范围内
        with torch.no_grad():
            threshold_params.clamp_(0.2, 0.8)
        
        logger.info(f"LoGoFair Post-Processing Epoch {pp_epoch + 1}/{post_processing_epochs}: "
                    f"Total Loss: {total_loss.item():.4f}, "
                    f"Thresholds: {threshold_params.cpu().detach().numpy()}")
    
    # 保存优化后的阈值
    final_thresholds = {0: threshold_params[0].item(), 1: threshold_params[1].item()}
    logger.info(f"LoGoFair final thresholds: {final_thresholds}")
    
    # 应用最终阈值进行评估
    final_calibrated_preds = all_preds_concat.clone()
    for g in [0, 1]:
        mask = (all_sensitive_concat == g)
        if mask.sum() > 0:
            threshold = final_thresholds[g]
            final_calibrated_preds[mask] = (all_preds_concat[mask] >= threshold).float()
    
    final_accuracy = (final_calibrated_preds == all_labels_concat.float()).sum() / len(all_labels_concat)
    final_dp, final_eo = compute_fairness_metrics(final_calibrated_preds, all_labels_concat, all_sensitive_concat)
    
    logger.info(f"LoGoFair Final Results - Accuracy: {final_accuracy:.4f}, DP: {final_dp:.4f}, EO: {final_eo:.4f}")
    
    # 保存全局模型
    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_LoGoFair.pt")
    torch.save(global_model, save_path)
    
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    
    return global_model, total_gpu_seconds, total_communication_cost
