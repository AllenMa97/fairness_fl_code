# PDFFed: Prototype Driven Fair Federated Learning
import copy
import os
import gc
import random
import time
import torch
import math
import numpy as np
import traceback
import torch.nn.functional as F


from tool.logger import *
from tool.utils import get_parameters, set_parameters, cos_sim, FL_fairness_and_accuracy_test_4_IMG_CLF,FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test
from hypothesis.generator import LatentGenerator, FigGenerator
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.client_parallel import ClientParallelExecutor
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def get_client_i_Prototype(param_dict, model, device, client_i_dataloader):
    model.to(device)

    time_cost = 0
    result_dict = {
        "client_i_label_0_prototype": None,
        "client_i_group_0_label_0_prototype": None,
        "client_i_group_1_label_0_prototype": None,
        "client_i_label_1_prototype": None,
        "client_i_group_0_label_1_prototype": None,
        "client_i_group_1_label_1_prototype": None,
    }
    client_i_label_0_feature_list = []
    client_i_group_0_label_0_feature_list = []
    client_i_group_1_label_0_feature_list = []
    client_i_label_1_feature_list = []
    client_i_group_0_label_1_feature_list = []
    client_i_group_1_label_1_feature_list = []

    with torch.no_grad():
        for batch_id, batch in enumerate(client_i_dataloader):
            # labels尺寸 [batch_size]
            labels = batch["labels"].to(device)
            # protected_label尺寸 [batch_size]
            protecteds = batch["protected"].to(device)
            # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
            true_batch_size = labels.size()[0]
            if "SENT_CLF" in param_dict["task"]:
                # input_ids尺寸 [batch_size, max_len]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)

            sent_label_flag = labels.gt(0.5)
            sent_group_flag = protecteds.gt(0.5)

            # 记录GPU计算开始时间
            gpu_start_time = time.time()

            if "SENT_CLF" in param_dict["task"]:
                # features尺寸 [batch_size, emb_dim]
                # logits尺寸 [batch_size, category]
                features, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # activated_preds = logits.softmax(dim=1)
                activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                _, preds = torch.max(activated_preds, dim=1)
                # batch_loss尺寸 [batch_size]

            elif "IMG_CLF" in param_dict["task"]:
                # preds尺寸 [batch_size, 1]
                # features尺寸 [batch_size, emb_dim]
                preds, features = model(imgs)

            elif "Tabular_CLF" in param_dict["task"]:
                # local_prediction尺寸 [batch_size, 1]
                if "ANN" in str(type(model)):
                    local_prediction, features = model(X)
                elif "LogisticRegression" in str(type(model)):
                    local_prediction = model(X)
                else:
                    local_prediction = model(X)


            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            time_cost += gpu_end_time - gpu_start_time

            client_i_label_0_feature_list.append(features[~sent_label_flag])
            client_i_group_0_label_0_feature_list.append(features[~sent_group_flag * ~sent_label_flag])
            client_i_group_1_label_0_feature_list.append(features[sent_group_flag * ~sent_label_flag])

            client_i_label_1_feature_list.append(features[sent_label_flag])
            client_i_group_0_label_1_feature_list.append(features[~sent_group_flag * sent_label_flag])
            client_i_group_1_label_1_feature_list.append(features[sent_group_flag * sent_label_flag])

        # Label 0
        if len(client_i_label_0_feature_list) != 0:
            client_i_label_0_prototype = torch.concatenate(client_i_label_0_feature_list, dim=0).mean(dim=0)
            result_dict['client_i_label_0_prototype'] = client_i_label_0_prototype
        # Label 0, Group 0
        if len(client_i_group_0_label_0_feature_list) != 0:
            client_i_group_0_label_0_prototype = torch.concatenate(client_i_group_0_label_0_feature_list, dim=0).mean(dim=0)
            result_dict['client_i_group_0_label_0_prototype'] = client_i_group_0_label_0_prototype
        # Label 0, Group 1
        if len(client_i_group_1_label_0_feature_list) != 0:
            client_i_group_1_label_0_prototype = torch.concatenate(client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)
            result_dict['client_i_group_1_label_0_prototype'] = client_i_group_1_label_0_prototype

        # Label 1
        if len(client_i_label_1_feature_list) != 0:
            client_i_label_1_prototype = torch.concatenate(client_i_label_1_feature_list, dim=0).mean(dim=0)
            result_dict['client_i_label_1_prototype'] = client_i_label_1_prototype
        # Label 1, Group 0
        if len(client_i_group_0_label_1_feature_list) != 0:
            client_i_group_0_label_1_prototype = torch.concatenate(client_i_group_0_label_1_feature_list, dim=0).mean(dim=0)
            result_dict['client_i_group_0_label_1_prototype'] = client_i_group_0_label_1_prototype
        # Label 1, Group 1
        if len(client_i_group_1_label_1_feature_list) != 0:
            client_i_group_1_label_1_prototype = torch.concatenate(client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
            result_dict['client_i_group_1_label_1_prototype'] = client_i_group_1_label_1_prototype

    return time_cost, result_dict

def get_cov_between_sensitive_attribute_and_prototype_decision_distance(weight_list, z_list, prototype_decision_distance):
    cov = 0
    z_bar = sum([weight_list[i] * z_list[i] for i in range(len(weight_list))])
    prototype_decision_distance_bar = sum(prototype_decision_distance) / len(prototype_decision_distance)
    for i in range(len(weight_list)):
        cov += weight_list[i]* (z_list[i] - z_bar) * (prototype_decision_distance[i] - prototype_decision_distance_bar)
    return cov

def _train_single_client_pdffed(client_id, device, model, param_dict,
                                 training_dataloaders, algorithm_epoch_T,
                                 accumulation_steps, use_amp, scaler, criterion,
                                 basic_path, iter_t, communication_round_I, num_clients_K,
                                 global_model,
                                 global_group_0_label_0_prototype_list,
                                 global_group_1_label_0_prototype_list,
                                 global_group_0_label_1_prototype_list,
                                 global_group_1_label_1_prototype_list):
    id = client_id
    client_i_aggregation_weight = param_dict.get('_client_aggregation_weights', {}).get(id, 1.0)

    # Local Initialization
    logger.info(f"Client {id} Init Local Model By Copy From Global Model")
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    client_i_dataloader = training_dataloaders[id]

    # Local Training
    logger.info("Start Local Training")
    users_gpu_seconds_list_item = 0
    for epoch in range(algorithm_epoch_T):
        # 设置状态变量
        epoch_total_loss = 0
        epoch_total_size = 0

        # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
        # FedAvg算法中，一个batch就更新一次参数
        for batch_id, batch in enumerate(client_i_dataloader):
            # labels尺寸 [batch_size]
            labels = batch["labels"].to(device)
            # protected_label尺寸 [batch_size]
            protecteds = batch["protected"]
            # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
            true_batch_size = labels.size()[0]
            epoch_total_size += true_batch_size
            if "SENT_CLF" in param_dict["task"]:
                # input_ids尺寸 [batch_size, max_len]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)

            # labels尺寸 [batch_size]
            labels = batch["labels"].to(device)
            # 记录GPU计算开始时间
            gpu_start_time = time.time()

            with autocast_context(device, use_amp):
                if "SENT_CLF" in param_dict["task"]:
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    # activated_preds尺寸 [batch_size, category]
                    features, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif "IMG_CLF" in param_dict["task"]:
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(model)):
                        local_prediction, features = model(X)
                    elif "LogisticRegression" in str(type(model)):
                        local_prediction = model(X)
                    else:
                        local_prediction = model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

            loss = torch.sum(batch_loss) / true_batch_size

            label_flag = labels.gt(0.5).float().reshape([-1, 1]).cpu()
            group_flag = protecteds.gt(0.5).float().reshape([-1, 1]).cpu()

            client_i_group_1_label_1_flag = (group_flag * label_flag)[:, 0].bool().tolist()
            client_i_group_0_label_1_flag = ((1 - group_flag) * label_flag)[:, 0].bool().tolist()
            client_i_group_1_label_0_flag = (group_flag * (1 - label_flag))[:, 0].bool().tolist()
            client_i_group_0_label_0_flag = ((1 - group_flag) * (1 - label_flag))[:, 0].bool().tolist()

            # 获取批内原型素材（保留梯度以影响模型训练）
            try:
                client_i_group_1_label_1_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],dim=0).to(device)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            try:
                client_i_group_0_label_1_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],dim=0).to(device)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            try:
                client_i_group_1_label_0_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],dim=0).to(device)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            try:
                client_i_group_0_label_0_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],dim=0).to(device)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            local_proto_weight_list = []
            local_proto_list = []
            local_proto_z_list = []
            local_proto_2_global_clf_label_list = []
            global_proto_list = []
            global_proto_2_local_clf_label_list = []

            # 以原型驱动的分类任务 作为 更新锚点
            # 局部原型 输入到局部分类器 的分类损失 # 局部原型 输入到全局分类器 的分类损失 # 全局原型 输入到局部分类器 的分类损失
            # 三类原型-分类器一致性损失初始化为 0.0 张量：即使本批无对应素材（如第1轮尚无全局原型），
            # 也能安全参与日志 .item() 与 loss 累加
            local_proto_2_local_clf_loss = torch.tensor(0.0, device=device)
            local_proto_2_global_clf_loss = torch.tensor(0.0, device=device)
            global_proto_2_local_clf_loss = torch.tensor(0.0, device=device)
            # 获取原型驱动的分类任务素材（保留梯度以影响模型训练）
            # Label 0, Group 0
            if "SENT_CLF" in param_dict["task"]:
                tmp_label = torch.tensor([1, 0]).float().to(device)
            elif "IMG_CLF" in param_dict["task"]:
                tmp_label = torch.zeros(1).to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                tmp_label = torch.zeros(1).to(device)
            if len(global_group_0_label_0_prototype_list) != 0:
                g = global_group_0_label_0_prototype_list[-1].to(device)
                global_proto_list.append(g)
                global_proto_2_local_clf_label_list.append(tmp_label)
            try:
                l = client_i_group_0_label_0_feature_in_one_batch.mean(dim=0).to(device)
                local_proto_list.append(l)
                local_proto_weight_list.append(sum(client_i_group_0_label_0_flag) / true_batch_size)
                local_proto_z_list.append(0)
                local_proto_2_global_clf_label_list.append(tmp_label)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            # Label 0, Group 1
            if "SENT_CLF" in param_dict["task"]:
                tmp_label = torch.tensor([1, 0]).float().to(device)
            elif "IMG_CLF" in param_dict["task"]:
                tmp_label = torch.zeros(1).to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                tmp_label = torch.zeros(1).to(device)
            if len(global_group_1_label_0_prototype_list) != 0:
                g = global_group_1_label_0_prototype_list[-1].to(device)
                global_proto_list.append(g)
                global_proto_2_local_clf_label_list.append(tmp_label)
            try:
                l = client_i_group_1_label_0_feature_in_one_batch.mean(dim=0).to(device)
                local_proto_list.append(l)
                local_proto_weight_list.append(sum(client_i_group_1_label_0_flag) / true_batch_size)
                local_proto_z_list.append(1)
                local_proto_2_global_clf_label_list.append(tmp_label)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            # Label 1, Group 0
            if "SENT_CLF" in param_dict["task"]:
                tmp_label = torch.tensor([0, 1]).float().to(device)
            elif "IMG_CLF" in param_dict["task"]:
                tmp_label = torch.ones(1).to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                tmp_label = torch.ones(1).to(device)
            if len(global_group_0_label_1_prototype_list) != 0:
                g = global_group_0_label_1_prototype_list[-1].to(device)
                global_proto_list.append(g)
                global_proto_2_local_clf_label_list.append(tmp_label)
            try:
                l = client_i_group_0_label_1_feature_in_one_batch.mean(dim=0).to(device)
                local_proto_list.append(l)
                local_proto_weight_list.append(sum(client_i_group_0_label_1_flag) / true_batch_size)
                local_proto_z_list.append(0)
                local_proto_2_global_clf_label_list.append(tmp_label)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            # Label 1, Group 1
            if "SENT_CLF" in param_dict["task"]:
                tmp_label = torch.tensor([0, 1]).float().to(device)
            elif "IMG_CLF" in param_dict["task"]:
                tmp_label = torch.ones(1).to(device)
            elif "Tabular_CLF" in param_dict["task"]:
                tmp_label = torch.ones(1).to(device)
            if len(global_group_1_label_1_prototype_list) != 0:
                g = global_group_1_label_1_prototype_list[-1].to(device)
                global_proto_list.append(g)
                global_proto_2_local_clf_label_list.append(tmp_label)
            try:
                l = client_i_group_1_label_1_feature_in_one_batch.mean(dim=0).to(device)
                local_proto_list.append(l)
                local_proto_weight_list.append(sum(client_i_group_1_label_1_flag) / true_batch_size)
                local_proto_z_list.append(1)
                local_proto_2_global_clf_label_list.append(tmp_label)
            except Exception:
                # 有异常则表示batch内没抽到这个group&这个label的数据
                pass

            local_proto_2_local_clf_decision_distance_list = []
            local_proto_tensors = torch.stack(local_proto_list).to(device)
            local_proto_2_global_clf_label_tensors = torch.stack(local_proto_2_global_clf_label_list).to(device)
            # 引理4：L_l2l 前对原型 stop-gradient（.detach()），确保 ∂L_l2l/∂φ ≡ 0，梯度仅进入本地分类器 w
            __, local_proto_2_local_clf_tmp_logit = model.only_clf_forward(local_proto_tensors.detach())
            if "SENT_CLF" in param_dict["task"]:
                max_logit_in_dim_0 = torch.max(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                min_logit_in_dim_0 = torch.min(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                max_logit_in_dim_1 = torch.max(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                min_logit_in_dim_1 = torch.min(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit.detach().clone()
                if max_logit_in_dim_0 == min_logit_in_dim_0:
                    normalized_local_proto_2_local_clf_tmp_logit[:, 0] = local_proto_2_local_clf_tmp_logit[:, 0]
                else:
                    normalized_local_proto_2_local_clf_tmp_logit[:, 0] = (local_proto_2_local_clf_tmp_logit[:, 0] - min_logit_in_dim_0) / (max_logit_in_dim_0 - min_logit_in_dim_0)

                if max_logit_in_dim_1 == min_logit_in_dim_1:
                    normalized_local_proto_2_local_clf_tmp_logit[:, 1] = local_proto_2_local_clf_tmp_logit[:, 1]
                else:
                    normalized_local_proto_2_local_clf_tmp_logit[:, 1] = (local_proto_2_local_clf_tmp_logit[:, 1] - min_logit_in_dim_1) / (max_logit_in_dim_1 - min_logit_in_dim_1)


            elif "IMG_CLF" in param_dict["task"]:
                max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                if max_logit == min_logit:
                    normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                else:
                    normalized_local_proto_2_local_clf_tmp_logit = (local_proto_2_local_clf_tmp_logit - min_logit) / (max_logit - min_logit)
            elif "Tabular_CLF" in param_dict["task"]:
                max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                if max_logit == min_logit:
                    normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                else:
                    normalized_local_proto_2_local_clf_tmp_logit = (local_proto_2_local_clf_tmp_logit - min_logit) / (max_logit - min_logit)

            if "SENT_CLF" in param_dict["task"]:
                decision_distance = normalized_local_proto_2_local_clf_tmp_logit[:,1] - normalized_local_proto_2_local_clf_tmp_logit[:, 0]
                local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
            elif "IMG_CLF" in param_dict["task"]:
                decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
            elif "Tabular_CLF" in param_dict["task"]:
                decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()

            del normalized_local_proto_2_local_clf_tmp_logit


            if torch.isnan(local_proto_2_local_clf_tmp_logit).any():
                logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
            else:
                local_proto_2_local_clf_loss += criterion(
                    local_proto_2_local_clf_tmp_logit.to(device),
                    local_proto_2_global_clf_label_tensors.to(device)
                ).mean()  # 局部原型 输入到局部分类器 的分类损失（梯度仅进本地分类器 w，引理4）

            global_model.to(device)
            # 引理6：L_l2g 的操作对象是原型 μ^L（不对全局分类器权重 w 施加约束）。
            # 常数化全局分类器参数（本地优化器本就不更新它），使梯度仅经输入原型路径进入本地表征模块 φ
            global_clf_grad_flags = [p.requires_grad for p in global_model.parameters()]
            for p in global_model.parameters():
                p.requires_grad_(False)
            __, local_proto_2_global_clf_tmp_logit = global_model.only_clf_forward(local_proto_tensors)
            for p, flag in zip(global_model.parameters(), global_clf_grad_flags):
                p.requires_grad_(flag)
            global_model.cpu()
            if torch.isnan(local_proto_2_global_clf_tmp_logit).any():
                logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
            else:
                local_proto_2_global_clf_loss += criterion(
                    local_proto_2_global_clf_tmp_logit.to(device),
                    local_proto_2_global_clf_label_tensors.to(device)
                ).mean()  # 局部原型 输入到全局分类器 的分类损失（保留梯度）
            if len(global_proto_2_local_clf_label_list) != 0:
                global_proto_tensors = torch.stack(global_proto_list).to(device)
                global_proto_2_local_clf_label_tensors = torch.stack(global_proto_2_local_clf_label_list).to(device)
                __, global_proto_2_local_clf_loss_tmp_logit = model.only_clf_forward(global_proto_tensors)
                if torch.isnan(global_proto_2_local_clf_loss_tmp_logit).any():
                    logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
                else:
                    global_proto_2_local_clf_loss += criterion(
                        global_proto_2_local_clf_loss_tmp_logit.to(device),
                        global_proto_2_local_clf_label_tensors.to(device)
                    ).mean()  # 全局原型 输入到局部分类器 的分类损失（全局原型无图，梯度进本地分类器 w，引理7）

            # 注：原 label_0/1_pred_distribution_gap（logit 空间群组差距）已移除。
            # 理由：该量是 Δ_rep 在 w 方向的投影（|w^T(μ_{g0,l}-μ_{g1,l})| ≤ |w|·||μ_{g0,l}-μ_{g1,l}||），
            # 已被 proto_alignment_loss（特征空间，约束全空间 Δ_rep，对应定理2）完全覆盖。
            # 公平性下界诊断由 delta_conf_loss（原型级，对应定理6，联动定理5）承担。

            # ===== 设计C：原型级群组置信度差异 δ_conf（对应理论证明.md 定理6）=====
            # 定理6（原型级）：δ_conf^(l) = |c(wᵀμ_{g0,l}+b) − c(wᵀμ_{g1,l}+b)|，
            #   低/高置信度设定下 EO ≥ Φ(2·δ_conf/σ_{g1}) − 0.5 − O(σ_z/σ_{g1})，
            #   公平性优化区间内线性化为 EO ≥ C·δ_conf − O(σ_z, Δ_Σ)。
            # 联动定理5（样本级）：Δ_c^(l) = |E[c|g=0,y=l] − E[c|g=1,y=l]| 大 ⟹ EO 下界大；
            #   由 c(s)=σ(|s|) 的 1/4-Lipschitz 性，δ_conf ≤ Δ_c + O(σ_z)，原型级度量继承样本级下界性质。
            # 含义：δ_conf 大 → EO 下界高 → 公平性问题严重
            #       最小化 δ_conf ⟂ 降低 EO 下界（必要条件方向）
            # 与特征空间原型对齐（定理2—PA收缩Δ_rep；定理3—PA约束EO上界）互补：
            # 原型对齐压 EO 上界，δ_conf 压 EO 下界，协同收窄 EO 允许区间。
            # 设计动机（由样本级改为原型级）：
            #   - 与 PDFFed 的 Prototype-Driven 命名一致，优化信号均可追溯到类-组原型；
            #   - 无需逐样本统计群组置信度，仅依赖本地已维护的类-组原型
            #     （client_i_group_*_label_*_feature_in_one_batch），进一步降低敏感信息暴露；
            #   - 在群组均值处评估，信号更稳定，规避小样本群组的置信度估计噪声。
            # 梯度友好处理：
            #   - 用两个 label 的置信度差距之和代替 min(abs(...))（smooth surrogate）
            #   - abs 用 |x| = √(x²+ε) 近似以保留梯度（ε=1e-8 避免数值问题）
            # 置信度定义（类-组原型过分类器，保留梯度）：
            #   SENT_CLF (2类): c = softmax(logit).max(dim=1)
            #   IMG/Tabular (单输出): c = sigmoid(|logit|) = max(sigmoid(logit), 1-sigmoid(logit))
            # 梯度作用域：δ_conf 前向走全模型（原型 → 分类头 → 置信度），但分类头参数被临时常数化，
            #   δ_conf的梯度仅经原型 μ_{g,l} 均值路径进入表征模块 φ，不扰动本地分类器 w
            #   （w 的校准职责由任务损失 / L_l2l / L_g2l 承担）。
            conf_epsilon = 1e-8
            delta_conf_loss = torch.tensor(0.0, device=device, requires_grad=False)
            # 常数化分类头参数后，only_clf_forward 的 logit 仍可对输入原型求导（梯度→φ），但对分类头权重不再求导（梯度→w 恒为零）。
            clf_head = None
            for module in model.children():
                if isinstance(module, torch.nn.Linear):
                    clf_head = module
            if clf_head is None:
                # 无法常数化分类头时直接跳过 δ_conf（置 0）：既避免其梯度污染分类头 w，
                # 也省去无谓的前向与建图开销（不浪费显存与算力）
                logger.info("### WARNING: 未找到分类头模块，跳过 δ_conf 计算（置 0）###")
            else:
                clf_head_grad_flags = [p.requires_grad for p in clf_head.parameters()]
                for p in clf_head.parameters():
                    p.requires_grad_(False)
                for l in [0, 1]:
                    g0_var = f"client_i_group_0_label_{l}_feature_in_one_batch"
                    g1_var = f"client_i_group_1_label_{l}_feature_in_one_batch"
                    # 批内未抽到该 (group, label) 时变量不存在，跳过该 label
                    if g0_var in locals() and g1_var in locals():
                        # 类-组原型 μ_{g,l}（批内特征均值，保留梯度）
                        proto_g0 = locals()[g0_var].mean(dim=0)
                        proto_g1 = locals()[g1_var].mean(dim=0)
                        # 原型过分类器 → 群组标准 logit（μ'_g = wᵀμ_{g,l}+b）
                        _, proto_g0_logit = model.only_clf_forward(proto_g0.unsqueeze(0))
                        _, proto_g1_logit = model.only_clf_forward(proto_g1.unsqueeze(0))
                        if "SENT_CLF" in param_dict["task"]:
                            c_g0 = torch.softmax(proto_g0_logit, dim=1).max(dim=1).values[0]
                            c_g1 = torch.softmax(proto_g1_logit, dim=1).max(dim=1).values[0]
                        else:  # IMG_CLF / Tabular_CLF：单输出 c = σ(|logit|)
                            c_g0 = torch.sigmoid(proto_g0_logit[:, 0].abs())[0]
                            c_g1 = torch.sigmoid(proto_g1_logit[:, 0].abs())[0]
                        diff = c_g0 - c_g1
                        # |x| ≈ √(x²+ε)，梯度友好
                        delta_conf_loss = delta_conf_loss + torch.sqrt(diff * diff + conf_epsilon)
                # 恢复分类头参数的 requires_grad（供任务损失 / L_l2l / L_g2l 正常求导）
                for p, flag in zip(clf_head.parameters(), clf_head_grad_flags):
                    p.requires_grad_(flag)

            # 保留 cov_abs 作为诊断量记录（不参与 loss），用于日志观察
            if len(local_proto_weight_list) > 0 and len(local_proto_2_local_clf_decision_distance_list) > 0:
                weight_tensor = torch.tensor(local_proto_weight_list, dtype=torch.float32, device=device)
                z_tensor = torch.tensor(local_proto_z_list, dtype=torch.float32, device=device)
                decision_distance_tensor = torch.tensor(local_proto_2_local_clf_decision_distance_list, dtype=torch.float32, device=device)
                z_bar = torch.sum(weight_tensor * z_tensor)
                decision_distance_bar = torch.mean(decision_distance_tensor)
                with torch.no_grad():
                    cov = torch.sum(weight_tensor * (z_tensor - z_bar) * (decision_distance_tensor - decision_distance_bar))
                    cov_abs = torch.abs(cov).item()
            else:
                cov_abs = 0.0

            # ===== 设计A：特征空间原型对齐损失（直接对应定理2）=====
            # proto_alignment_loss = 0.5 * Σ_l ||μ_{g0,l} - μ_{g1,l}||²
            # 定理2 证明了其 Δ_rep 收缩性：Δ_rep^(t+1) = (1−2η)·Δ_rep^(t)
            #   Δ_rep^(t+T) ≤ Δ_rep^(t) - η·λ·T·γ,  γ = E[||μ_{g0,l} - μ_{g1,l}||] > 0
            # 与上方 label_X_pred_distribution_gap（logit空间，仅 w 方向投影）互补：
            #   logit 差距 = |w^T(μ_{g1,l} - μ_{g0,l})| ≤ |w|·||μ_{g1,l} - μ_{g0,l}||
            # 特征空间 proto alignment 约束全空间 Δ_rep，logit 空间约束投影分量，两者不冗余。
            proto_alignment_loss = torch.tensor(0.0, device=device)
            # Label 0 的群组原型配对
            if ('client_i_group_0_label_0_feature_in_one_batch' in locals() and
                'client_i_group_1_label_0_feature_in_one_batch' in locals()):
                proto_g0_l0 = client_i_group_0_label_0_feature_in_one_batch.mean(dim=0)
                proto_g1_l0 = client_i_group_1_label_0_feature_in_one_batch.mean(dim=0)
                proto_alignment_loss = proto_alignment_loss + 0.5 * torch.norm(proto_g0_l0 - proto_g1_l0, p=2) ** 2
            # Label 1 的群组原型配对
            if ('client_i_group_0_label_1_feature_in_one_batch' in locals() and
                'client_i_group_1_label_1_feature_in_one_batch' in locals()):
                proto_g0_l1 = client_i_group_0_label_1_feature_in_one_batch.mean(dim=0)
                proto_g1_l1 = client_i_group_1_label_1_feature_in_one_batch.mean(dim=0)
                proto_alignment_loss = proto_alignment_loss + 0.5 * torch.norm(proto_g0_l1 - proto_g1_l1, p=2) ** 2

            # 5项reg：
            # proto_alignment_loss对应理论分析里面的L_PA（定理2，前向后向均仅影响编码器 φ）
            # delta_conf_loss对应理论分析里面的L_Conf（定理6，前向走全模型，梯度仅进表征模块）
            # local_proto_2_local_clf_loss对应理论分析里面的L_l2l，梯度仅进本地分类器 w（引理4，原型 stop-gradient，∂L/∂φ ≡ 0）

            #   L_l2g  梯度仅进本地表征模块 φ（引理6，操作对象是原型 μ^L，不约束全局分类器 w）
            #   L_g2l  梯度进本地分类器 w（引理7，全局原型无图，天然只动 w）
            lamda_list = [1, 1, 1, 1, 1]
            reg_list = [
                proto_alignment_loss,
                delta_conf_loss,
                local_proto_2_local_clf_loss,
                local_proto_2_global_clf_loss,
                global_proto_2_local_clf_loss,
            ]
            if float(batch_id) % 1 == 0:
            # if iter_t != 0 and float(batch_id) % 10 == 0:
                logger.info(f"### Origin task loss：{loss.item()} ;\n"
                            f"proto_alignment_loss (L_PA): {round(proto_alignment_loss.item(), 5)} ;\n"
                            f"delta_conf_loss (L_Conf): {round(delta_conf_loss.item(), 5)} ;\n"
                            f"local_proto_2_local_clf_loss (L_l2l): {round(local_proto_2_local_clf_loss.item(), 5)} ;\n"
                            f"local_proto_2_global_clf_loss (L_l2g): {round(local_proto_2_global_clf_loss.item(), 5)} ;\n"
                            f"global_proto_2_local_clf_loss (L_g2l): {round(global_proto_2_local_clf_loss.item(), 5)} ;\n"
                            f"cov_abs(diag): {round(cov_abs if isinstance(cov_abs, float) else cov_abs.item(), 5)} ;\n"
                            f"in Batch_id:{batch_id} of Epoch:{epoch} in Client:{id}. ### ")
            for index, lamda in enumerate(lamda_list):
                loss += lamda * reg_list[index]

            scale_backward(loss, scaler)
            if (batch_id + 1) % accumulation_steps == 0:
                # FedAvg算法一个batch就做一次更新
                scaler_step(scaler, optimizer)
                # 清空梯度
                model.zero_grad()

            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list_item += (gpu_end_time - gpu_start_time)

            # 记录状态信息
            epoch_total_loss += loss
            # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
            #     client_datasets_size_list[id] / param_dict['batch_size'])

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, labels, batch_loss, loss
            elif "IMG_CLF" in param_dict["task"]:
                del imgs, labels, batch_loss, loss

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

        # logger.debug(f"GPU Memory :")
        # logger.debug(torch.cuda.memory_summary())
        # torch.cuda.empty_cache()
        gc.collect()


    # Upgrade the local model list
    client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
    # local_model_list[id] = model.cpu()  # 内存化
    torch.save(model.cpu(), client_model_path)  # 持久化

    # 记录GPU计算开始时间
    gpu_start_time = time.time()

    # 计算客户的 类原型
    # logger.info("~~~~~~~~~~~~~ 5. 计算客户的 类原型 ~~~~~~~~~~~~~~~")
    time_cost, result_dict = get_client_i_Prototype(param_dict, model, device, client_i_dataloader)

    # Prototype accumulation
    proto_g0_l0 = None
    proto_g1_l0 = None
    proto_g0_l1 = None
    proto_g1_l1 = None
    weighted_proto_g0_l0 = None
    weighted_proto_g1_l0 = None
    weighted_proto_g0_l1 = None
    weighted_proto_g1_l1 = None

    with torch.no_grad():
        # Label 0, Group 0
        if result_dict['client_i_group_0_label_0_prototype'] is not None:
            client_i_group_0_label_0_prototype = result_dict['client_i_group_0_label_0_prototype']
            proto_g0_l0 = client_i_group_0_label_0_prototype
            weighted_proto_g0_l0 = client_i_aggregation_weight * client_i_group_0_label_0_prototype

        # Label 0, Group 1
        if result_dict['client_i_group_1_label_0_prototype'] is not None:
            client_i_group_1_label_0_prototype = result_dict['client_i_group_1_label_0_prototype']
            proto_g1_l0 = client_i_group_1_label_0_prototype
            weighted_proto_g1_l0 = client_i_aggregation_weight * client_i_group_1_label_0_prototype

        # Label 1, Group 0
        if result_dict['client_i_group_0_label_1_prototype'] is not None:
            client_i_group_0_label_1_prototype = result_dict['client_i_group_0_label_1_prototype']
            proto_g0_l1 = client_i_group_0_label_1_prototype
            weighted_proto_g0_l1 = client_i_aggregation_weight * client_i_group_0_label_1_prototype

        # Label 1, Group 1
        if result_dict['client_i_group_1_label_1_prototype'] is not None:
            client_i_group_1_label_1_prototype = result_dict['client_i_group_1_label_1_prototype']
            proto_g1_l1 = client_i_group_1_label_1_prototype
            weighted_proto_g1_l1 = client_i_aggregation_weight * client_i_group_1_label_1_prototype

    # 记录GPU计算结束时间
    gpu_end_time = time.time()
    users_gpu_seconds_list_item += (gpu_end_time - gpu_start_time)

    # ===== 挑战1：差分隐私噪声注入（对应定理5，可选）=====
    # 定理5：客户端在上传原型前添加 ε-LDP 噪声 ξ ~ N(0, σ_noise²·I_d)，
    #   σ_noise² ≥ 2d·ln(2/δ)/ε²，则聚合后满足分布式差分隐私。
    # 仅当 param_dict['use_dp']=True 时启用，否则不影响主算法。
    # 本地训练不受影响——噪声只加在待上传的原型上。
    if param_dict.get('use_dp', False):
        epsilon = param_dict.get('dp_epsilon', 1.0)
        dp_delta = param_dict.get('dp_delta', 1e-5)
        # 原型维度 d（从已有原型推断，若无原型则跳过）
        proto_dim = None
        for p in [proto_g0_l0, proto_g1_l0, proto_g0_l1, proto_g1_l1]:
            if p is not None:
                proto_dim = p.shape[0]
                break
        if proto_dim is not None:
            # 定理5 的方差下界：σ_noise² ≥ 2d·ln(2/δ)/ε²
            sigma_noise_sq = 2.0 * proto_dim * math.log(2.0 / dp_delta) / (epsilon ** 2)
            sigma_noise = math.sqrt(sigma_noise_sq)
            # 对每个待上传原型加噪（加权原型同样加噪，保证聚合后仍满足 DP）
            with torch.no_grad():
                if weighted_proto_g0_l0 is not None:
                    weighted_proto_g0_l0 = weighted_proto_g0_l0 + torch.randn_like(weighted_proto_g0_l0) * sigma_noise
                if weighted_proto_g1_l0 is not None:
                    weighted_proto_g1_l0 = weighted_proto_g1_l0 + torch.randn_like(weighted_proto_g1_l0) * sigma_noise
                if weighted_proto_g0_l1 is not None:
                    weighted_proto_g0_l1 = weighted_proto_g0_l1 + torch.randn_like(weighted_proto_g0_l1) * sigma_noise
                if weighted_proto_g1_l1 is not None:
                    weighted_proto_g1_l1 = weighted_proto_g1_l1 + torch.randn_like(weighted_proto_g1_l1) * sigma_noise
                # 非加权原型也加噪（用于 Server 端后训练的全局原型）
                if proto_g0_l0 is not None:
                    proto_g0_l0 = proto_g0_l0 + torch.randn_like(proto_g0_l0) * sigma_noise
                if proto_g1_l0 is not None:
                    proto_g1_l0 = proto_g1_l0 + torch.randn_like(proto_g1_l0) * sigma_noise
                if proto_g0_l1 is not None:
                    proto_g0_l1 = proto_g0_l1 + torch.randn_like(proto_g0_l1) * sigma_noise
                if proto_g1_l1 is not None:
                    proto_g1_l1 = proto_g1_l1 + torch.randn_like(proto_g1_l1) * sigma_noise
            logger.info(f"Client {id}: DP noise injected (ε={epsilon}, δ={dp_delta}, "
                        f"d={proto_dim}, σ_noise={round(sigma_noise, 5)})")

    del model
    gc.collect()
    # torch.cuda.empty_cache()

    return {
        'gpu_seconds': users_gpu_seconds_list_item,
        'client_i_aggregation_weight': client_i_aggregation_weight,
        'weighted_prototypes': {
            'group_0_label_0': weighted_proto_g0_l0,
            'group_1_label_0': weighted_proto_g1_l0,
            'group_0_label_1': weighted_proto_g0_l1,
            'group_1_label_1': weighted_proto_g1_l1,
        },
        'prototypes': {
            'group_0_label_0': proto_g0_l0,
            'group_1_label_0': proto_g1_l0,
            'group_0_label_1': proto_g0_l1,
            'group_1_label_1': proto_g1_l1,
        },
    }


def PDF_Fed(device,
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
    accumulation_steps = int(256 / param_dict['batch_size'])
    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)

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

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    
    # 记录开始时间
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    num_of_class = 2
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

    if "SENT_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024*1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out.parameters()) * 4 / (1024*1024)

    elif "IMG_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out_layer.parameters()) * 4 / (1024*1024)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")


    # ===== 设计E：EO_proto 自适应校准步数（对总轮次完全免疫）=====
    # 定理2：EO ≥ 2·δ_conf - O(σ_z, Δ_Σ)，δ_conf（此处用 EO_proto 代理）
    # 大 → EO 下界高 → 需要更多校准力度；小 → 1 步即可维持。
    # 第 1 轮建立 baseline，后续每轮步数 N = 1 + 1{EO_proto ≥ baseline}。
    # 即：EO_proto 未低于 baseline 时多走 1 步，已低于 baseline 时只走 1 步。
    # 对 5 轮或 500 轮场景均适用，无需可调参数。
    eo_baseline = None  # 第 1 轮的 EO_proto，作为自适应基准

    global_group_0_label_0_prototype_list = []
    global_group_1_label_0_prototype_list = []
    global_group_0_label_1_prototype_list = []
    global_group_1_label_1_prototype_list = []

    prototype_gap_threshold = -99999 # gap一开始很小。若本地的gap比全局的gap要小，证明局部的表征已经很贴近全局了，则不需要传表征参数

    accumulated_Communication_Cost = 0

    parallel_executor = ClientParallelExecutor(
        device=device,
        global_model=global_model,
        param_dict=param_dict,
        needs_global_model_during_training=True,  # PDFFed uses global_model.only_clf_forward during training
    )

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(start_round, communication_round_I):
        users_gpu_seconds_list = [0] * num_clients_K

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

        selected_client_training_dataset_size = sum([client_datasets_size_list[item] for item in idxs_users])
        average_weight = [0 for _ in range(num_clients_K)]
        for id in idxs_users:
            average_weight[id] = client_datasets_size_list[id] / selected_client_training_dataset_size
        average_weight = np.array(average_weight)

        accumulated_Communication_Cost += len(idxs_users) * model_MB_size
        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        global_group_0_label_0_feature_list = []
        global_group_1_label_0_feature_list = []
        global_group_0_label_1_feature_list = []
        global_group_1_label_1_feature_list = []

        weighted_global_group_0_label_0_feature_list = []
        weighted_global_group_1_label_0_feature_list = []
        weighted_global_group_0_label_1_feature_list = []
        weighted_global_group_1_label_1_feature_list = []

        prototype_gap_between_client_i_and_global_list = []
        weighted_prototype_gap_between_client_i_and_global_list = []

        # Simulate Client Parallel
        # Pass aggregation weights to the training function via param_dict
        param_dict['_client_aggregation_weights'] = {id: average_weight[id] for id in idxs_users}

        results = parallel_executor.run_clients(
            idxs_users,
            _train_single_client_pdffed,
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
            global_model=global_model,
            global_group_0_label_0_prototype_list=global_group_0_label_0_prototype_list,
            global_group_1_label_0_prototype_list=global_group_1_label_0_prototype_list,
            global_group_0_label_1_prototype_list=global_group_0_label_1_prototype_list,
            global_group_1_label_1_prototype_list=global_group_1_label_1_prototype_list,
        )

        # Collect results
        for i, id in enumerate(idxs_users):
            users_gpu_seconds_list[id] += results[i]['gpu_seconds']
            # Accumulate prototypes from results
            r = results[i]
            if r['prototypes']['group_0_label_0'] is not None:
                global_group_0_label_0_feature_list.append(r['prototypes']['group_0_label_0'])
                weighted_global_group_0_label_0_feature_list.append(r['weighted_prototypes']['group_0_label_0'])
            if r['prototypes']['group_1_label_0'] is not None:
                global_group_1_label_0_feature_list.append(r['prototypes']['group_1_label_0'])
                weighted_global_group_1_label_0_feature_list.append(r['weighted_prototypes']['group_1_label_0'])
            if r['prototypes']['group_0_label_1'] is not None:
                global_group_0_label_1_feature_list.append(r['prototypes']['group_0_label_1'])
                weighted_global_group_0_label_1_feature_list.append(r['weighted_prototypes']['group_0_label_1'])
            if r['prototypes']['group_1_label_1'] is not None:
                global_group_1_label_1_feature_list.append(r['prototypes']['group_1_label_1'])
                weighted_global_group_1_label_1_feature_list.append(r['weighted_prototypes']['group_1_label_1'])

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # Global operation

        # 更新全局原型
        logger.info("Prototype aggregation update")
        (global_group_0_label_0_prototype, global_group_0_label_1_prototype) = 0, 0
        (global_group_1_label_0_prototype, global_group_1_label_1_prototype) = 0, 0

        # 前面已经乘过权重（client_i_aggregation_weight）了，所以这里只需要加起来即可得到全局的prototype
        # Label 0, Group 0
        if len(weighted_global_group_0_label_0_feature_list) != 0:
            for proto in weighted_global_group_0_label_0_feature_list:
                global_group_0_label_0_prototype += proto
            # 全局原型更新：直接以"加权客户端原型的累加和"作为本轮全局原型，并追加历史（供回溯）。非EMA。
            if len(global_group_0_label_0_prototype_list) != 0:
                global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)
            else:
                global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
        # Label 0, Group 1
        if len(weighted_global_group_1_label_0_feature_list) != 0:
            for proto in weighted_global_group_1_label_0_feature_list:
                global_group_1_label_0_prototype += proto
            # 全局原型更新：直接以"加权客户端原型的累加和"作为本轮全局原型，并追加历史（供回溯）。非EMA。
            if len(global_group_1_label_0_prototype_list) != 0:
                global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)
            else:
                global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
        # Label 1, Group 0
        if len(weighted_global_group_0_label_1_feature_list) != 0:
            for proto in weighted_global_group_0_label_1_feature_list:
                global_group_0_label_1_prototype += proto
            # 全局原型更新：直接以"加权客户端原型的累加和"作为本轮全局原型，并追加历史（供回溯）。非EMA。
            if len(global_group_0_label_1_prototype_list) != 0:
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)
            else:
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
        # Label 1, Group 1
        if len(weighted_global_group_1_label_1_feature_list) != 0:
            for proto in weighted_global_group_1_label_1_feature_list:
                global_group_1_label_1_prototype += proto
            # 全局原型更新：直接以"加权客户端原型的累加和"作为本轮全局原型，并追加历史（供回溯）。非EMA。
            if len(global_group_1_label_1_prototype_list) != 0:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)
            else:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []

        # 读取正常客户的参数
        theta_list = []
        rep_theta_list = []

        aggregation_weights = []
        rep_aggregation_weights = []

        # 获取参数聚合的素材
        for index, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化

            if "SENT_CLF" in param_dict["task"]:
                rep_model = selected_model.bert
            elif "IMG_CLF" in param_dict["task"]:
                rep_model = selected_model.shared_base
            elif "Tabular_CLF" in param_dict["task"]:
                rep_model = selected_model.shared_base

            param = get_parameters(selected_model)
            theta_list.append(param)

            # 计算该客户端的更新量
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(param, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            rep_theta_start_index, rep_theta_end_index= 0, len(get_parameters(rep_model))
            rep_theta_list.append(param[rep_theta_start_index : rep_theta_end_index])
            aggregation_weights.append(client_datasets_size_list[id]) # 这个地方只需要读取客户的数据量，不用除以总量！
            rep_aggregation_weights.append(client_datasets_size_list[id]) # 这个地方只需要读取客户的数据量，不用除以总量！

            del selected_model
            gc.collect()

        # 参数聚合
        try:
            if (len(aggregation_weights) != 0) and (sum(aggregation_weights) != 0):
                logger.info("Parameter aggregation")
                # 聚合完整的参数
                theta_list = np.array(theta_list, dtype=object)
                # FedAvg旧版论文的聚合权重是平均
                # theta_avg = np.mean(theta_list, 0).tolist()
                # FedAvg新版论文的聚合权重是数据占比
                # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的"分子"，不用传整个分数，"分母"会自动除
                # 如一个weights = [w1, w2, w3, w4]
                # 那么结果就是(theta1 * w1 + theta2 * w2 + theta3 * w3 + theta4 * w4)/ sum(w1+w2+w3+w4)
                theta_avg = np.average(theta_list, axis=0, weights=aggregation_weights).tolist()

                # 聚合表征模块的参数
                rep_theta_list = np.array(rep_theta_list, dtype=object)
                # 如果sum(rep_aggregation_weights)为0，那么所有参与方都没上传表征模块，不用再替换全局参数
                if sum(rep_aggregation_weights) != 0:
                    # 用rep_aggregation_weights的权重聚合Rep模块
                    rep_theta_list_avg = np.average(rep_theta_list, axis=0, weights=rep_aggregation_weights).tolist()
                    # 把表征部分的参数替换回去
                    # 之前尝试过rep和clf分开处理，而不是现在这种替换，但是会有类型转换问题
                    theta_avg[rep_theta_start_index : rep_theta_end_index] = rep_theta_list_avg

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)

                del theta_list
                gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

        logger.info(f"Communication Round {(iter_t + 1)}  Communication Cost: {accumulated_Communication_Cost} MB")

        logger.info("Testing before post training")
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

        # Server side post training
        logger.info("Server side post training")
        global_model.to(device)
        Server_side_post_training_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        if "SENT_CLF" in param_dict["task"]:
            Server_side_post_training_optimizer.set_parameters(list(global_model.out.named_parameters()))
        elif "IMG_CLF" in param_dict["task"]:
            Server_side_post_training_optimizer.set_parameters(list(global_model.out_layer.named_parameters()))
        elif "Tabular_CLF" in param_dict["task"]:
            Server_side_post_training_optimizer.set_parameters(list(global_model.out_layer.named_parameters()))

        post_training_feature_group_label_list = []
        # Label 0, Group 0
        if len(weighted_global_group_0_label_0_feature_list) != 0:
            post_training_feature_group_label_list.append((global_group_0_label_0_prototype, 0, 0))
        # Label 0, Group 1
        if len(weighted_global_group_1_label_0_feature_list) != 0:
            post_training_feature_group_label_list.append((global_group_1_label_0_prototype, 1, 0))
        # Label 1, Group 0
        if len(weighted_global_group_0_label_1_feature_list) != 0:
            post_training_feature_group_label_list.append((global_group_0_label_1_prototype, 0, 1))
        # Label 1, Group 1
        if len(weighted_global_group_1_label_1_feature_list) != 0:
            post_training_feature_group_label_list.append((global_group_1_label_1_prototype, 1, 1))

        # ===== 设计F：Server端后训练双目标 L_post = L_cls + λ_eo · EO_proto =====
        # 定理4 已证明 EO_proto 是 EO_global 的有效代理：
        #   (a) |EO_proto - EO_global| ≤ ε(Δ_rep, Δ_Σ)
        #   (b) cos⟨∇EO_proto, ∇EO_global⟩ ≥ 1 - ε
        #   (c) EO_proto 下降 ⟹ EO_global 下降
        # 原实现仅用 L_cls（精度目标），现在补上 EO_proto（公平性目标）。
        # λ_eo 默认为 1.0，可通过 param_dict['lambda_eo'] 调节。
        # 同时计算当前 EO_proto 值（供设计E 的反弹触发使用）。
        lambda_eo = param_dict.get('lambda_eo', 1.0)
        # 按标签组织原型，便于配对计算 EO_proto
        # proto_by_label[l] = {g0: prototype_tensor, g1: prototype_tensor}
        proto_by_label = {0: {}, 1: {}}
        for (proto, g, l) in post_training_feature_group_label_list:
            proto_by_label[l][g] = proto

        # 计算当前 EO_proto（用于设计E 的反弹诊断，no_grad）
        current_EO_proto = 0.0
        with torch.no_grad():
            for l in [0, 1]:
                if 0 in proto_by_label[l] and 1 in proto_by_label[l]:
                    x_g0 = proto_by_label[l][0].to(device)
                    x_g1 = proto_by_label[l][1].to(device)
                    _, logit_g0 = global_model.only_clf_forward(x_g0)
                    _, logit_g1 = global_model.only_clf_forward(x_g1)
                    if "SENT_CLF" in param_dict["task"]:
                        # 2分类：σ(正类logit) = softmax[1]
                        p_g0 = torch.softmax(logit_g0, dim=0)[1]
                        p_g1 = torch.softmax(logit_g1, dim=0)[1]
                    else:
                        p_g0 = torch.sigmoid(logit_g0)
                        p_g1 = torch.sigmoid(logit_g1)
                    current_EO_proto += abs(p_g0 - p_g1).item()
        logger.info(f"Current EO_proto (global prototypes): {round(current_EO_proto, 5)}")

        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        for item in post_training_feature_group_label_list:
            x = item[0].to(device)
            if item[2] == 1:
                if "SENT_CLF" in param_dict["task"]:
                    tmp_label = torch.tensor([0, 1]).float().to(device)
                elif "IMG_CLF" in param_dict["task"]:
                    tmp_label = torch.ones(1).to(device)
                elif "Tabular_CLF" in param_dict["task"]:
                    tmp_label = torch.ones(1).to(device)
            else:
                if "SENT_CLF" in param_dict["task"]:
                    tmp_label = torch.tensor([1, 0]).float().to(device)
                elif "IMG_CLF" in param_dict["task"]:
                    tmp_label = torch.zeros(1).to(device)
                elif "Tabular_CLF" in param_dict["task"]:
                    tmp_label = torch.zeros(1).to(device)
            with autocast_context(device, use_amp):
                __, tmp_logit = global_model.only_clf_forward(x)
            if torch.isnan(tmp_logit).any():
                logger.info("### The tmp_logit is nan in Server side post training ###")
            else:
                # 精度目标 L_cls
                L_cls = criterion(tmp_logit.to(device), tmp_label.to(device))
                post_training_loss = torch.sum(L_cls)
                scale_backward(post_training_loss, scaler)
                scaler_step(scaler, Server_side_post_training_optimizer)

        # ===== 设计E + 设计F：EO_proto 自适应步数公平性校准 =====
        # 设计F：EO_proto 下降步（公平性目标，对应定理4）
        # 设计E：步数 N 自适应（对应定理2 的下界诊断）
        #   定理2：EO ≥ 2·δ_conf - O(σ_z, Δ_Σ)
        #   EO_proto（δ_conf 代理）大 → EO 下界高 → 需要更多校准步
        #   EO_proto 小 → 下界已低 → 1 步即可维持
        # 第 1 轮建立 baseline，后续 N = 1 + 1{EO_proto ≥ baseline}
        # （未低于 baseline 时多走 1 步，已低于 baseline 时只走 1 步）
        # 对 5 轮或 500 轮场景均适用，无魔术数字。
        if eo_baseline is None and current_EO_proto > 0:
            eo_baseline = current_EO_proto
            logger.info(f"### EO_proto baseline established: {round(eo_baseline, 5)} ###")
        if eo_baseline is not None and current_EO_proto >= eo_baseline:
            eo_calibration_steps = 2  # 未低于 baseline，多校准 1 步
        else:
            eo_calibration_steps = 1  # 已低于 baseline，维持 1 步
        logger.info(f"EO_proto adaptive calibration: current={round(current_EO_proto, 5)}, "
                    f"baseline={round(eo_baseline, 5) if eo_baseline is not None else 'N/A'}, "
                    f"steps={eo_calibration_steps}")

        # EO_proto = Σ_l |σ(w^T μ_{g0,l}+b) - σ(w^T μ_{g1,l}+b)|
        # 用 √(x²+ε) 代理 abs 以保留梯度
        eo_epsilon = 1e-8
        for _ in range(eo_calibration_steps):
            for l in [0, 1]:
                if 0 in proto_by_label[l] and 1 in proto_by_label[l]:
                    x_g0 = proto_by_label[l][0].to(device)
                    x_g1 = proto_by_label[l][1].to(device)
                    with autocast_context(device, use_amp):
                        _, logit_g0 = global_model.only_clf_forward(x_g0)
                        _, logit_g1 = global_model.only_clf_forward(x_g1)
                        if "SENT_CLF" in param_dict["task"]:
                            p_g0 = torch.softmax(logit_g0, dim=0)[1]
                            p_g1 = torch.softmax(logit_g1, dim=0)[1]
                        else:
                            p_g0 = torch.sigmoid(logit_g0)
                            p_g1 = torch.sigmoid(logit_g1)
                        diff = p_g0 - p_g1
                        EO_proto_item = torch.sqrt(diff * diff + eo_epsilon)
                    eo_loss = lambda_eo * EO_proto_item
                    scale_backward(eo_loss, scaler)
                    scaler_step(scaler, Server_side_post_training_optimizer)

        # 记录GPU计算结束时间
        gpu_end_time = time.time()
        total_gpu_seconds += (gpu_end_time - gpu_start_time)

        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                                   testing_dataset_len)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
                
                # ===== TensorBoard logging =====
                log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=accumulated_Communication_Cost)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=accumulated_Communication_Cost,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist())
                flush()
                
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                
                # ===== TensorBoard logging =====
                log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=accumulated_Communication_Cost)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=accumulated_Communication_Cost,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist())
                flush()
                
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                
                # ===== TensorBoard logging =====
                log_test_metrics(accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                    FR=float(FR), HM=float(HM),
                    step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                    communication_cost=accumulated_Communication_Cost)
                log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds,
                    communication_cost=accumulated_Communication_Cost,
                    selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist())
                flush()


        # ===== 深度监控（每轮都执行，包括最后一轮）=====
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, 
                         iter_t + 1, client_model_updates=client_model_updates)

        # 保存检查点（按 checkpoint_save_freq 间隔，包含原型信息）
        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time,
                extra_state={
                    'global_group_0_label_0_prototype_list': [p.cpu().tolist() for p in global_group_0_label_0_prototype_list] if global_group_0_label_0_prototype_list else [],
                    'global_group_1_label_0_prototype_list': [p.cpu().tolist() for p in global_group_1_label_0_prototype_list] if global_group_1_label_0_prototype_list else [],
                    'global_group_0_label_1_prototype_list': [p.cpu().tolist() for p in global_group_0_label_1_prototype_list] if global_group_0_label_1_prototype_list else [],
                    'global_group_1_label_1_prototype_list': [p.cpu().tolist() for p in global_group_1_label_1_prototype_list] if global_group_1_label_1_prototype_list else [],
                    'accumulated_Communication_Cost': accumulated_Communication_Cost
                }
            )

            # 清理旧检查点，保留最近 N 个
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_PDFFed.pt")
    torch.save(global_model, save_path)

    total_communication_cost = accumulated_Communication_Cost
    return global_model, total_gpu_seconds, total_communication_cost
