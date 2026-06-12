import copy
import os
import gc
import time
import torch
import numpy as np
import torch.nn.functional as F


from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import get_parameters, set_parameters, cos_sim, get_HM_by_two_value, FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"



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


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# FedAvg+FedProx的采样方法

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


def PDF_Fed_V2(device,
             global_model,
             algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
             training_dataloaders,
             training_dataset,
             client_dataset_list,
             param_dict,
             testing_dataloader,
             testing_dataset_len
             ):
    logger.info("!!!!!!!!!!!!   PDF_Fed_V2(   !!!!!!!!!!!!!!!!")

    accumulation_steps = int(256 / param_dict['batch_size'])

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

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

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    num_of_class = 2
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

    if "SENT_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out.parameters()) * 4 / (1024 * 1024)

    elif "IMG_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out_layer.parameters()) * 4 / (1024 * 1024)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # 自定义初始参数
    # try:
    #     EMA_frac = param_dict['EMA_frac']
    # except Exception:
    #     EMA_frac = 0.1

    EMA_frac = 0  # 相当于不使用EMA

    global_group_0_label_0_prototype_list = []
    global_group_1_label_0_prototype_list = []
    global_group_0_label_1_prototype_list = []
    global_group_1_label_1_prototype_list = []

    prototype_gap_threshold = -99999  # gap一开始很小。若本地的gap比全局的gap要小，证明局部的表征已经很贴近全局了，则不需要传表征参数

    accumulated_Communication_Cost = 0

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        users_gpu_seconds_list = [0] * num_clients_K

        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            # style="FedAvg",
            style="FedProx",
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
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            # 下发模型
            logger.info(f"Client {id} Init Local Model By Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            # Local Training
            logger.info("Start Local Training")
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

                    # 获取批内原型素材
                    # logger.info("#### 获取批内原型素材 #####")
                    with torch.no_grad():
                        try:
                            client_i_group_1_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).to(device)
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
                    # 局部原型 输入到局部分类器 的分类损失 # 局部原型 输入到全局分类器 的分类损失
                    local_proto_2_local_clf_loss, local_proto_2_global_clf_loss = 0, 0
                    # 获取原型驱动的分类任务素材
                    with torch.no_grad():
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
                            tmp_label = torch.zeros(1).to(device)
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
                    __, local_proto_2_local_clf_tmp_logit = model.only_clf_forward(local_proto_tensors)
                    if "SENT_CLF" in param_dict["task"]:
                        max_logit_in_dim_0 = torch.max(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        min_logit_in_dim_0 = torch.min(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        max_logit_in_dim_1 = torch.max(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        min_logit_in_dim_1 = torch.min(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit.detach().clone()
                        if max_logit_in_dim_0 == min_logit_in_dim_0:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = local_proto_2_local_clf_tmp_logit[:, 0]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 0] - min_logit_in_dim_0) / (
                                                                                             max_logit_in_dim_0 - min_logit_in_dim_0)

                        if max_logit_in_dim_1 == min_logit_in_dim_1:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = local_proto_2_local_clf_tmp_logit[:, 1]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 1] - min_logit_in_dim_1) / (
                                                                                             max_logit_in_dim_1 - min_logit_in_dim_1)


                    elif "IMG_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)
                    elif "Tabular_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)

                    if "SENT_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit[:, 1] - \
                                            normalized_local_proto_2_local_clf_tmp_logit[:, 0]
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "IMG_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "Tabular_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()

                    del normalized_local_proto_2_local_clf_tmp_logit
                    gc.collect()
                    # torch.cuda.empty_cache()

                    if torch.isnan(local_proto_2_local_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
                    else:
                        local_proto_2_local_clf_loss += criterion(
                            local_proto_2_local_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到局部分类器 的分类损失

                    global_model.to(device)
                    __, local_proto_2_global_clf_tmp_logit = global_model.only_clf_forward(local_proto_tensors)
                    global_model.cpu()
                    if torch.isnan(local_proto_2_global_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_global_clf_tmp_logit ###")
                    else:
                        local_proto_2_global_clf_loss += criterion(
                            local_proto_2_global_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到全局分类器 的分类损失
                    

                    # 群组决策差距（在本地增强群组公平性）
                    # 标签0和1 不同群组的预测分布差距
                    label_0_pred_distribution_gap, label_1_pred_distribution_gap = 0, 0
                    if "SENT_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "IMG_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "Tabular_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                    # 敏感属性 与 原型决策边界距离 的协方差

                    cov = get_cov_between_sensitive_attribute_and_prototype_decision_distance(
                        weight_list=local_proto_weight_list, z_list=local_proto_z_list,
                        prototype_decision_distance=local_proto_2_local_clf_decision_distance_list)
                    # print(f"cov: {cov}")
                    cov_abs = abs(cov)

                    lamda_list = [1, 1, 
                                  1, 1,
                                  1]  # FedPro思路
                    reg_list = [
                        local_proto_2_local_clf_loss, local_proto_2_global_clf_loss,
                        label_0_pred_distribution_gap, label_1_pred_distribution_gap,
                        cov_abs
                    ]
                    if float(batch_id) % 50 == 0:
                        # if iter_t != 0 and float(batch_id) % 10 == 0:
                        logger.info(f"### Origin task loss：{loss.item()} ;\n"
                                    f"local_proto_2_local_clf_loss：{round(local_proto_2_local_clf_loss, 5)} ;\n"
                                    f"local_proto_2_global_clf_loss: {round(local_proto_2_global_clf_loss, 5)} ;\n"

                                    f"label_0_pred_distribution_gap：{round(label_0_pred_distribution_gap, 5)} ;\n"
                                    f"label_1_pred_distribution_gap: {round(label_1_pred_distribution_gap, 5)} ;\n"

                                    f"cov_abs: {round(cov_abs, 5)} ;\n"

                                    f"in Batch_id:{batch_id} of Epoch:{epoch} in Client:{id}. ### ")
                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * reg_list[index]

                    # del sent_label_flag, sent_group_flag
                    # del client_i_group_1_label_1_feature_in_one_batch, client_i_group_0_label_1_feature_in_one_batch
                    # del client_i_group_1_label_0_feature_in_one_batch, client_i_group_0_label_0_feature_in_one_batch

                    loss.backward()
                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()
                        # 清空梯度
                        model.zero_grad()

                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels, batch_loss, loss
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels, batch_loss, loss

                    gc.collect()
                    # torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

                # logger.debug(f"GPU Memory :")
                # logger.debug(torch.cuda.memory_summary())
                # # torch.cuda.empty_cache()
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

            with torch.no_grad():
                # Label 0, Group 0
                if result_dict['client_i_group_0_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_0_prototype = result_dict['client_i_group_0_label_0_prototype']
                    global_group_0_label_0_feature_list.append(client_i_group_0_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_0_prototype)

                # Label 0, Group 1
                if result_dict['client_i_group_1_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_0_prototype = result_dict['client_i_group_1_label_0_prototype']
                    global_group_1_label_0_feature_list.append(client_i_group_1_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_0_prototype)

                # Label 1, Group 0
                if result_dict['client_i_group_0_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_1_prototype = result_dict['client_i_group_0_label_1_prototype']
                    global_group_0_label_1_feature_list.append(client_i_group_0_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_1_prototype)

                # Label 1, Group 1
                if result_dict['client_i_group_1_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_1_prototype = result_dict['client_i_group_1_label_1_prototype']
                    global_group_1_label_1_feature_list.append(client_i_group_1_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_1_prototype)

            # del model
            # gc.collect()
            # # torch.cuda.empty_cache()
            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            del model
            gc.collect()
            # torch.cuda.empty_cache()

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
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_0_prototype_list) != 0:
                global_group_0_label_0_prototype_list.append(
                    EMA_frac * global_group_0_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_0_prototype
                )
            else:
                global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
        # Label 0, Group 1
        if len(weighted_global_group_1_label_0_feature_list) != 0:
            for proto in weighted_global_group_1_label_0_feature_list:
                global_group_1_label_0_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_0_prototype_list) != 0:
                global_group_1_label_0_prototype_list.append(
                    EMA_frac * global_group_1_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_0_prototype
                )
            else:
                global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
        # Label 1, Group 0
        if len(weighted_global_group_0_label_1_feature_list) != 0:
            for proto in weighted_global_group_0_label_1_feature_list:
                global_group_0_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_1_prototype_list) != 0:
                global_group_0_label_1_prototype_list.append(
                    EMA_frac * global_group_0_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_1_prototype
                )
            else:
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
        # Label 1, Group 1
        if len(weighted_global_group_1_label_1_feature_list) != 0:
            for proto in weighted_global_group_1_label_1_feature_list:
                global_group_1_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_1_prototype_list) != 0:
                global_group_1_label_1_prototype_list.append(
                    EMA_frac * global_group_1_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_1_prototype
                )
            else:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

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
            rep_theta_start_index, rep_theta_end_index = 0, len(get_parameters(rep_model))
            rep_theta_list.append(param[rep_theta_start_index: rep_theta_end_index])
            aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！
            rep_aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

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
                # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
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
                    theta_avg[rep_theta_start_index: rep_theta_end_index] = rep_theta_list_avg

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)

                del theta_list
                gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

        logger.info(f"Communication Round {(iter_t + 1)}  Communication Cost: {accumulated_Communication_Cost} MB")

        logger.info("Testing before post training")
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                               testing_dataset_len)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader,
                                                                         testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

        # Server side post training
        logger.info("Server side post training")
        global_model.to(device)
        Server_side_post_training_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                                                learning_rate=param_dict['learning_rate'],
                                                                max_grad_norm=0)
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
            __, tmp_logit = global_model.only_clf_forward(x)
            if torch.isnan(tmp_logit).any():
                logger.info("### The tmp_logit is nan in Server side post training ###")
            else:
                post_training_loss = criterion(tmp_logit.to(device), tmp_label.to(device))
                post_training_loss = torch.sum(post_training_loss)
                post_training_loss.backward()
                Server_side_post_training_optimizer.step()

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
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_PDFFed.pt")
    torch.save(global_model, save_path)

    total_communication_cost = accumulated_Communication_Cost
    return global_model, total_gpu_seconds, total_communication_cost



def PDF_Fed_V2_Prox_Client_Sampling(device,
                                 global_model,
                                 algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                                 training_dataloaders,
                                 training_dataset,
                                 client_dataset_list,
                                 param_dict,
                                 testing_dataloader,
                                 testing_dataset_len
                                 ):
    logger.info("!!!!!!!!!!!!   PDF_Fed_V2_Prox_Client_Sampling   !!!!!!!!!!!!!!!!")

    accumulation_steps = int(256 / param_dict['batch_size'])

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

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

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    num_of_class = 2
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

    if "SENT_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out.parameters()) * 4 / (1024 * 1024)

    elif "IMG_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out_layer.parameters()) * 4 / (1024 * 1024)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # 自定义初始参数
    # try:
    #     EMA_frac = param_dict['EMA_frac']
    # except Exception:
    #     EMA_frac = 0.1

    EMA_frac = 0  # 相当于不使用EMA

    global_group_0_label_0_prototype_list = []
    global_group_1_label_0_prototype_list = []
    global_group_0_label_1_prototype_list = []
    global_group_1_label_1_prototype_list = []

    prototype_gap_threshold = -99999  # gap一开始很小。若本地的gap比全局的gap要小，证明局部的表征已经很贴近全局了，则不需要传表征参数

    accumulated_Communication_Cost = 0

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        users_gpu_seconds_list = [0] * num_clients_K

        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            # style="FedAvg",
            style="FedProx",
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
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            # 下发模型
            logger.info(f"Client {id} Init Local Model By Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            # Local Training
            logger.info("Start Local Training")
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

                    # 获取批内原型素材
                    # logger.info("#### 获取批内原型素材 #####")
                    with torch.no_grad():
                        try:
                            client_i_group_1_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).to(device)
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
                    # 局部原型 输入到局部分类器 的分类损失 # 局部原型 输入到全局分类器 的分类损失
                    local_proto_2_local_clf_loss, local_proto_2_global_clf_loss = 0, 0
                    # 获取原型驱动的分类任务素材
                    with torch.no_grad():
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
                            tmp_label = torch.zeros(1).to(device)
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
                    __, local_proto_2_local_clf_tmp_logit = model.only_clf_forward(local_proto_tensors)
                    if "SENT_CLF" in param_dict["task"]:
                        max_logit_in_dim_0 = torch.max(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        min_logit_in_dim_0 = torch.min(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        max_logit_in_dim_1 = torch.max(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        min_logit_in_dim_1 = torch.min(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit.detach().clone()
                        if max_logit_in_dim_0 == min_logit_in_dim_0:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = local_proto_2_local_clf_tmp_logit[:, 0]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 0] - min_logit_in_dim_0) / (
                                                                                             max_logit_in_dim_0 - min_logit_in_dim_0)

                        if max_logit_in_dim_1 == min_logit_in_dim_1:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = local_proto_2_local_clf_tmp_logit[:, 1]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 1] - min_logit_in_dim_1) / (
                                                                                             max_logit_in_dim_1 - min_logit_in_dim_1)


                    elif "IMG_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)
                    elif "Tabular_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)

                    if "SENT_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit[:, 1] - \
                                            normalized_local_proto_2_local_clf_tmp_logit[:, 0]
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "IMG_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "Tabular_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()

                    del normalized_local_proto_2_local_clf_tmp_logit
                    gc.collect()
                    # torch.cuda.empty_cache()

                    if torch.isnan(local_proto_2_local_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
                    else:
                        local_proto_2_local_clf_loss += criterion(
                            local_proto_2_local_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到局部分类器 的分类损失

                    global_model.to(device)
                    __, local_proto_2_global_clf_tmp_logit = global_model.only_clf_forward(local_proto_tensors)
                    global_model.cpu()
                    if torch.isnan(local_proto_2_global_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_global_clf_tmp_logit ###")
                    else:
                        local_proto_2_global_clf_loss += criterion(
                            local_proto_2_global_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到全局分类器 的分类损失
                    

                    # 群组决策差距（在本地增强群组公平性）
                    # 标签0和1 不同群组的预测分布差距
                    label_0_pred_distribution_gap, label_1_pred_distribution_gap = 0, 0
                    if "SENT_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "IMG_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "Tabular_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                    # 敏感属性 与 原型决策边界距离 的协方差

                    cov = get_cov_between_sensitive_attribute_and_prototype_decision_distance(
                        weight_list=local_proto_weight_list, z_list=local_proto_z_list,
                        prototype_decision_distance=local_proto_2_local_clf_decision_distance_list)
                    # print(f"cov: {cov}")
                    cov_abs = abs(cov)

                    lamda_list = [1, 1, 
                                  1, 1,
                                  1]  # FedPro思路
                    reg_list = [
                        local_proto_2_local_clf_loss, local_proto_2_global_clf_loss,
                        label_0_pred_distribution_gap, label_1_pred_distribution_gap,
                        cov_abs
                    ]
                    if float(batch_id) % 50 == 0:
                        # if iter_t != 0 and float(batch_id) % 10 == 0:
                        logger.info(f"### Origin task loss：{loss.item()} ;\n"
                                    f"local_proto_2_local_clf_loss：{round(local_proto_2_local_clf_loss, 5)} ;\n"
                                    f"local_proto_2_global_clf_loss: {round(local_proto_2_global_clf_loss, 5)} ;\n"

                                    f"label_0_pred_distribution_gap：{round(label_0_pred_distribution_gap, 5)} ;\n"
                                    f"label_1_pred_distribution_gap: {round(label_1_pred_distribution_gap, 5)} ;\n"

                                    f"cov_abs: {round(cov_abs, 5)} ;\n"

                                    f"in Batch_id:{batch_id} of Epoch:{epoch} in Client:{id}. ### ")
                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * reg_list[index]

                    # del sent_label_flag, sent_group_flag
                    # del client_i_group_1_label_1_feature_in_one_batch, client_i_group_0_label_1_feature_in_one_batch
                    # del client_i_group_1_label_0_feature_in_one_batch, client_i_group_0_label_0_feature_in_one_batch

                    loss.backward()
                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()
                        # 清空梯度
                        model.zero_grad()

                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels, batch_loss, loss
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels, batch_loss, loss

                    gc.collect()
                    # torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

                # logger.debug(f"GPU Memory :")
                # logger.debug(torch.cuda.memory_summary())
                # # torch.cuda.empty_cache()
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

            with torch.no_grad():
                # Label 0, Group 0
                if result_dict['client_i_group_0_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_0_prototype = result_dict['client_i_group_0_label_0_prototype']
                    global_group_0_label_0_feature_list.append(client_i_group_0_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_0_prototype)

                # Label 0, Group 1
                if result_dict['client_i_group_1_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_0_prototype = result_dict['client_i_group_1_label_0_prototype']
                    global_group_1_label_0_feature_list.append(client_i_group_1_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_0_prototype)

                # Label 1, Group 0
                if result_dict['client_i_group_0_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_1_prototype = result_dict['client_i_group_0_label_1_prototype']
                    global_group_0_label_1_feature_list.append(client_i_group_0_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_1_prototype)

                # Label 1, Group 1
                if result_dict['client_i_group_1_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_1_prototype = result_dict['client_i_group_1_label_1_prototype']
                    global_group_1_label_1_feature_list.append(client_i_group_1_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_1_prototype)

            # del model
            # gc.collect()
            # # torch.cuda.empty_cache()
            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            del model
            gc.collect()
            # torch.cuda.empty_cache()

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
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_0_prototype_list) != 0:
                global_group_0_label_0_prototype_list.append(
                    EMA_frac * global_group_0_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_0_prototype
                )
            else:
                global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
        # Label 0, Group 1
        if len(weighted_global_group_1_label_0_feature_list) != 0:
            for proto in weighted_global_group_1_label_0_feature_list:
                global_group_1_label_0_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_0_prototype_list) != 0:
                global_group_1_label_0_prototype_list.append(
                    EMA_frac * global_group_1_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_0_prototype
                )
            else:
                global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
        # Label 1, Group 0
        if len(weighted_global_group_0_label_1_feature_list) != 0:
            for proto in weighted_global_group_0_label_1_feature_list:
                global_group_0_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_1_prototype_list) != 0:
                global_group_0_label_1_prototype_list.append(
                    EMA_frac * global_group_0_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_1_prototype
                )
            else:
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
        # Label 1, Group 1
        if len(weighted_global_group_1_label_1_feature_list) != 0:
            for proto in weighted_global_group_1_label_1_feature_list:
                global_group_1_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_1_prototype_list) != 0:
                global_group_1_label_1_prototype_list.append(
                    EMA_frac * global_group_1_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_1_prototype
                )
            else:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

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
            rep_theta_start_index, rep_theta_end_index = 0, len(get_parameters(rep_model))
            rep_theta_list.append(param[rep_theta_start_index: rep_theta_end_index])
            aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！
            rep_aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

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
                # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
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
                    theta_avg[rep_theta_start_index: rep_theta_end_index] = rep_theta_list_avg

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)

                del theta_list
                gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

        logger.info(f"Communication Round {(iter_t + 1)}  Communication Cost: {accumulated_Communication_Cost} MB")

        logger.info("Testing before post training")
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                               testing_dataset_len)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader,
                                                                         testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

        # Server side post training
        logger.info("Server side post training")
        global_model.to(device)
        Server_side_post_training_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                                                learning_rate=param_dict['learning_rate'],
                                                                max_grad_norm=0)
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
            __, tmp_logit = global_model.only_clf_forward(x)
            if torch.isnan(tmp_logit).any():
                logger.info("### The tmp_logit is nan in Server side post training ###")
            else:
                post_training_loss = criterion(tmp_logit.to(device), tmp_label.to(device))
                post_training_loss = torch.sum(post_training_loss)
                post_training_loss.backward()
                Server_side_post_training_optimizer.step()

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
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_PDFFed.pt")
    torch.save(global_model, save_path)

    total_communication_cost = accumulated_Communication_Cost
    return global_model, total_gpu_seconds, total_communication_cost


def PDF_Fed_V2_RepModel_DynamicTrans(device,
                                  global_model,
                                  algorithm_epoch_T, num_clients_K, communication_round_I,
                                  FL_fraction, FL_drop_rate,
                                  training_dataloaders,
                                  training_dataset,
                                  client_dataset_list,
                                  param_dict,
                                  testing_dataloader,
                                  testing_dataset_len):
    logger.info("!!!!!!!!!!!!   PDF_Fed_V2_RepModel_DynamicTrans   !!!!!!!!!!!!!!!!")
    counter_RepModelTrans = 0  # 记录传输了RepModel的次数
    counter_no_RepModelTrans = 0  # 记录无传输RepModel的次数

    accumulation_steps = int(256 / param_dict['batch_size'])

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

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

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    num_of_class = 2
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

    if "SENT_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out.parameters()) * 4 / (1024 * 1024)

    elif "IMG_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out_layer.parameters()) * 4 / (1024 * 1024)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # 自定义初始参数
    # try:
    #     EMA_frac = param_dict['EMA_frac']
    # except Exception:
    #     EMA_frac = 0.1

    EMA_frac = 0  # 相当于不使用EMA

    global_group_0_label_0_prototype_list = []
    global_group_1_label_0_prototype_list = []
    global_group_0_label_1_prototype_list = []
    global_group_1_label_1_prototype_list = []

    prototype_gap_threshold = -99999  # gap一开始很小。若本地的gap比全局的gap要小，证明局部的表征已经很贴近全局了，则不需要传表征参数

    accumulated_Communication_Cost = 0

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        users_gpu_seconds_list = [0] * num_clients_K

        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            # style="FedAvg",
            style="FedProx",
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
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            # 下发模型
            logger.info(f"Client {id} Init Local Model By Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            # Local Training
            logger.info("Start Local Training")
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

                    # 获取批内原型素材
                    # logger.info("#### 获取批内原型素材 #####")
                    with torch.no_grad():
                        try:
                            client_i_group_1_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).to(device)
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
                    # 局部原型 输入到局部分类器 的分类损失 # 局部原型 输入到全局分类器 的分类损失
                    local_proto_2_local_clf_loss, local_proto_2_global_clf_loss = 0, 0
                    # 获取原型驱动的分类任务素材
                    with torch.no_grad():
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
                            tmp_label = torch.zeros(1).to(device)
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
                    __, local_proto_2_local_clf_tmp_logit = model.only_clf_forward(local_proto_tensors)
                    if "SENT_CLF" in param_dict["task"]:
                        max_logit_in_dim_0 = torch.max(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        min_logit_in_dim_0 = torch.min(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        max_logit_in_dim_1 = torch.max(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        min_logit_in_dim_1 = torch.min(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit.detach().clone()
                        if max_logit_in_dim_0 == min_logit_in_dim_0:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = local_proto_2_local_clf_tmp_logit[:, 0]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 0] - min_logit_in_dim_0) / (
                                                                                         max_logit_in_dim_0 - min_logit_in_dim_0)

                        if max_logit_in_dim_1 == min_logit_in_dim_1:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = local_proto_2_local_clf_tmp_logit[:, 1]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 1] - min_logit_in_dim_1) / (
                                                                                         max_logit_in_dim_1 - min_logit_in_dim_1)
                    elif "IMG_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                   local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                   max_logit - min_logit)
                    elif "Tabular_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                   local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                   max_logit - min_logit)

                    if "SENT_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit[:, 1] - \
                                            normalized_local_proto_2_local_clf_tmp_logit[:, 0]
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "IMG_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "Tabular_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()

                    del normalized_local_proto_2_local_clf_tmp_logit
                    gc.collect()
                    # torch.cuda.empty_cache()

                    if torch.isnan(local_proto_2_local_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
                    else:
                        local_proto_2_local_clf_loss += criterion(
                            local_proto_2_local_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到局部分类器 的分类损失

                    global_model.to(device)
                    __, local_proto_2_global_clf_tmp_logit = global_model.only_clf_forward(local_proto_tensors)
                    global_model.cpu()
                    if torch.isnan(local_proto_2_global_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_global_clf_tmp_logit ###")
                    else:
                        local_proto_2_global_clf_loss += criterion(
                            local_proto_2_global_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到全局分类器 的分类损失
                    

                    # 群组决策差距（在本地增强群组公平性）
                    # 标签0和1 不同群组的预测分布差距
                    label_0_pred_distribution_gap, label_1_pred_distribution_gap = 0, 0
                    if "SENT_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "IMG_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "Tabular_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                    # 敏感属性 与 原型决策边界距离 的协方差

                    cov = get_cov_between_sensitive_attribute_and_prototype_decision_distance(
                        weight_list=local_proto_weight_list, z_list=local_proto_z_list,
                        prototype_decision_distance=local_proto_2_local_clf_decision_distance_list)
                    # print(f"cov: {cov}")
                    cov_abs = abs(cov)

                    lamda_list = [1, 1, 
                                  1, 1,
                                  1]  # FedPro思路
                    reg_list = [
                        local_proto_2_local_clf_loss, local_proto_2_global_clf_loss,
                        label_0_pred_distribution_gap, label_1_pred_distribution_gap,
                        cov_abs
                    ]
                    if float(batch_id) % 50 == 0:
                        # if iter_t != 0 and float(batch_id) % 10 == 0:
                        logger.info(f"### Origin task loss：{loss.item()} ;\n"
                                    f"local_proto_2_local_clf_loss：{round(local_proto_2_local_clf_loss, 5)} ;\n"
                                    f"local_proto_2_global_clf_loss: {round(local_proto_2_global_clf_loss, 5)} ;\n"

                                    f"label_0_pred_distribution_gap：{round(label_0_pred_distribution_gap, 5)} ;\n"
                                    f"label_1_pred_distribution_gap: {round(label_1_pred_distribution_gap, 5)} ;\n"

                                    f"cov_abs: {round(cov_abs, 5)} ;\n"

                                    f"in Batch_id:{batch_id} of Epoch:{epoch} in Client:{id}. ### ")
                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * reg_list[index]

                    # del sent_label_flag, sent_group_flag
                    # del client_i_group_1_label_1_feature_in_one_batch, client_i_group_0_label_1_feature_in_one_batch
                    # del client_i_group_1_label_0_feature_in_one_batch, client_i_group_0_label_0_feature_in_one_batch

                    loss.backward()
                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()
                        # 清空梯度
                        model.zero_grad()

                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels, batch_loss, loss
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels, batch_loss, loss

                    # gc.collect()
                    # torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

                # logger.debug(f"GPU Memory :")
                # logger.debug(torch.cuda.memory_summary())
                # torch.cuda.empty_cache()
                # gc.collect()

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            # 记录GPU计算开始时间
            gpu_start_time = time.time()

            # 计算客户的 类原型
            # logger.info("~~~~~~~~~~~~~ 5. 计算客户的 类原型 ~~~~~~~~~~~~~~~")
            time_cost, result_dict = get_client_i_Prototype(param_dict, model, device, client_i_dataloader)
            # logger.info("~~~~~~~~~~~~~ 5. (用全局模型）计算客户的 类原型 ~~~~~~~~~~~~~~~")
            __, result_dict_from_global_model = get_client_i_Prototype(param_dict, global_model, device,
                                                                       client_i_dataloader)

            global_client_i_prototype_gap = 0
            with torch.no_grad():
                # Label 0, Group 0
                if result_dict['client_i_group_0_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_0_prototype = result_dict['client_i_group_0_label_0_prototype']

                    if result_dict_from_global_model['client_i_group_0_label_0_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_0_label_0_prototype']
                        g = result_dict_from_global_model['client_i_group_0_label_0_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_0_label_0_feature_list.append(client_i_group_0_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_0_prototype)

                # Label 0, Group 1
                if result_dict['client_i_group_1_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_0_prototype = result_dict['client_i_group_1_label_0_prototype']

                    if result_dict_from_global_model['client_i_group_1_label_0_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_1_label_0_prototype']
                        g = result_dict_from_global_model['client_i_group_1_label_0_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_1_label_0_feature_list.append(client_i_group_1_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_0_prototype)

                # Label 1, Group 0
                if result_dict['client_i_group_0_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_1_prototype = result_dict['client_i_group_0_label_1_prototype']

                    if result_dict_from_global_model['client_i_group_0_label_1_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_0_label_1_prototype']
                        g = result_dict_from_global_model['client_i_group_0_label_1_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_0_label_1_feature_list.append(client_i_group_0_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_1_prototype)

                # Label 1, Group 1
                if result_dict['client_i_group_1_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_1_prototype = result_dict['client_i_group_1_label_1_prototype']

                    if result_dict_from_global_model['client_i_group_1_label_1_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_1_label_1_prototype']
                        g = result_dict_from_global_model['client_i_group_1_label_1_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_1_label_1_feature_list.append(client_i_group_1_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_1_prototype)

            # del model
            # gc.collect()
            # # torch.cuda.empty_cache()
            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            # del model
            # gc.collect()
            # torch.cuda.empty_cache()

            prototype_gap_between_client_i_and_global_list.append(global_client_i_prototype_gap)
            # 注意这里的权重，不能用当前数据量占据整体数据的比例，要用参数聚合的权重（当前客户数据量占据所抽到的客户数据量总和的比例）
            # weighted_prototype_gap_between_client_i_and_global = (client_datasets_size_list[id] / training_dataset_size) * prototype_gap_between_client_i_and_global
            weighted_prototype_gap_between_client_i_and_global = client_i_aggregation_weight * global_client_i_prototype_gap
            weighted_prototype_gap_between_client_i_and_global_list.append(
                weighted_prototype_gap_between_client_i_and_global)

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
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_0_prototype_list) != 0:
                global_group_0_label_0_prototype_list.append(
                    EMA_frac * global_group_0_label_0_prototype_list[-1] + (
                            1 - EMA_frac) * global_group_0_label_0_prototype
                )
            else:
                global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
        # Label 0, Group 1
        if len(weighted_global_group_1_label_0_feature_list) != 0:
            for proto in weighted_global_group_1_label_0_feature_list:
                global_group_1_label_0_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_0_prototype_list) != 0:
                global_group_1_label_0_prototype_list.append(
                    EMA_frac * global_group_1_label_0_prototype_list[-1] + (
                            1 - EMA_frac) * global_group_1_label_0_prototype
                )
            else:
                global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
        # Label 1, Group 0
        if len(weighted_global_group_0_label_1_feature_list) != 0:
            for proto in weighted_global_group_0_label_1_feature_list:
                global_group_0_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_1_prototype_list) != 0:
                global_group_0_label_1_prototype_list.append(
                    EMA_frac * global_group_0_label_1_prototype_list[-1] + (
                            1 - EMA_frac) * global_group_0_label_1_prototype
                )
            else:
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
        # Label 1, Group 1
        if len(weighted_global_group_1_label_1_feature_list) != 0:
            for proto in weighted_global_group_1_label_1_feature_list:
                global_group_1_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_1_prototype_list) != 0:
                global_group_1_label_1_prototype_list.append(
                    EMA_frac * global_group_1_label_1_prototype_list[-1] + (
                            1 - EMA_frac) * global_group_1_label_1_prototype
                )
            else:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

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
            rep_theta_start_index, rep_theta_end_index = 0, len(get_parameters(rep_model))
            rep_theta_list.append(param[rep_theta_start_index: rep_theta_end_index])
            aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

            if prototype_gap_between_client_i_and_global_list[index] < prototype_gap_threshold:
                logger.info(
                    f"#@!#@!#@! Client {id}'s gap:{round(prototype_gap_between_client_i_and_global_list[index], 4)}, threshold:{round(prototype_gap_threshold, 4)} ;  ~ #@!#@!#@!")

                logger.info(
                    f"#@!#@!#@! The prototype_gap_between_client_{id}_and_global is too close, the Rep param not upload;  ~ #@!#@!#@!")
                counter_no_RepModelTrans += 1
                rep_aggregation_weights.append(0)
            else:
                logger.info(
                    f"#@!#@!#@! Client {id}'s gap:{round(prototype_gap_between_client_i_and_global_list[index], 4)}, threshold:{round(prototype_gap_threshold, 4)} ;  ~ #@!#@!#@!")

                counter_RepModelTrans += 1
                rep_aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

            # del selected_model
            # gc.collect()

        # 参数聚合
        try:
            if (len(aggregation_weights) != 0) and (sum(aggregation_weights) != 0):
                logger.info("Parameter aggregation")
                # 聚合完整的参数
                theta_list = np.array(theta_list, dtype=object)
                # FedAvg旧版论文的聚合权重是平均
                # theta_avg = np.mean(theta_list, 0).tolist()
                # FedAvg新版论文的聚合权重是数据占比
                # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
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
                    theta_avg[rep_theta_start_index: rep_theta_end_index] = rep_theta_list_avg

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)

                # del theta_list
                # gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

        logger.info(f"Communication Round {(iter_t + 1)}  Communication Cost: {accumulated_Communication_Cost} MB")

        # 更新全局原型Gap
        if len(weighted_prototype_gap_between_client_i_and_global_list) != 0:
            prototype_gap_threshold = np.array(weighted_prototype_gap_between_client_i_and_global_list).mean()

        logger.info("Testing before post training")
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                               testing_dataset_len)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader,
                                                                         testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

        # Server side post training
        logger.info("Server side post training")
        global_model.to(device)
        Server_side_post_training_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                                                learning_rate=param_dict['learning_rate'],
                                                                max_grad_norm=0)
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
            __, tmp_logit = global_model.only_clf_forward(x)
            if torch.isnan(tmp_logit).any():
                logger.info("### The tmp_logit is nan in Server side post training ###")
            else:
                post_training_loss = criterion(tmp_logit.to(device), tmp_label.to(device))
                post_training_loss = torch.sum(post_training_loss)
                post_training_loss.backward()
                Server_side_post_training_optimizer.step()

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
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

    logger.info("Training finish, save and return the global model.")
    logger.info(
        f"Rep Model Trans Time: {counter_RepModelTrans}, not trans time:{counter_no_RepModelTrans}, total:{counter_RepModelTrans + counter_no_RepModelTrans}")
    logger.info(
        f"Rep Model Trans Rate: {counter_RepModelTrans / (counter_RepModelTrans + counter_no_RepModelTrans)}, not trans Rate:{counter_no_RepModelTrans / (counter_RepModelTrans + counter_no_RepModelTrans)}")

    # Save global model
    # save_dir = f'./save_path/'
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"global_PDFFed.pt")
    # torch.save(global_model, save_path)

    total_communication_cost = accumulated_Communication_Cost
    return global_model, total_gpu_seconds, total_communication_cost


def PDF_Fed_V2_RepModel_DynamicTrans_Prox_Client_Sampling(device,
                                                       global_model,
                                                       algorithm_epoch_T, num_clients_K, communication_round_I,
                                                       FL_fraction, FL_drop_rate,
                                                       training_dataloaders,
                                                       training_dataset,
                                                       client_dataset_list,
                                                       param_dict,
                                                       testing_dataloader,
                                                       testing_dataset_len
                                                       ):
    logger.info("!!!!!!!!!!!!   PDF_Fed_V2_RepModel_DynamicTrans_Prox_Client_Sampling   !!!!!!!!!!!!!!!!")
    counter_RepModelTrans = 0  # 记录传输了RepModel的次数
    counter_no_RepModelTrans = 0  # 记录无传输RepModel的次数

    accumulation_steps = int(256 / param_dict['batch_size'])

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

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

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    num_of_class = 2
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

    if "SENT_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out.parameters()) * 4 / (1024 * 1024)

    elif "IMG_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out_layer.parameters()) * 4 / (1024 * 1024)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # 自定义初始参数
    # try:
    #     EMA_frac = param_dict['EMA_frac']
    # except Exception:
    #     EMA_frac = 0.1

    EMA_frac = 0  # 相当于不使用EMA

    global_group_0_label_0_prototype_list = []
    global_group_1_label_0_prototype_list = []
    global_group_0_label_1_prototype_list = []
    global_group_1_label_1_prototype_list = []

    prototype_gap_threshold = -99999  # gap一开始很小。若本地的gap比全局的gap要小，证明局部的表征已经很贴近全局了，则不需要传表征参数

    accumulated_Communication_Cost = 0

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        users_gpu_seconds_list = [0] * num_clients_K

        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            # style="FedAvg",
            style="FedProx",
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
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            # 下发模型
            logger.info(f"Client {id} Init Local Model By Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            # Local Training
            logger.info("Start Local Training")
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

                    # 获取批内原型素材
                    # logger.info("#### 获取批内原型素材 #####")
                    with torch.no_grad():
                        try:
                            client_i_group_1_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).to(device)
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
                    # 局部原型 输入到局部分类器 的分类损失 # 局部原型 输入到全局分类器 的分类损失
                    local_proto_2_local_clf_loss, local_proto_2_global_clf_loss = 0, 0
                    # 获取原型驱动的分类任务素材
                    with torch.no_grad():
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
                            tmp_label = torch.zeros(1).to(device)
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
                    __, local_proto_2_local_clf_tmp_logit = model.only_clf_forward(local_proto_tensors)
                    if "SENT_CLF" in param_dict["task"]:
                        max_logit_in_dim_0 = torch.max(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        min_logit_in_dim_0 = torch.min(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        max_logit_in_dim_1 = torch.max(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        min_logit_in_dim_1 = torch.min(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit.detach().clone()
                        if max_logit_in_dim_0 == min_logit_in_dim_0:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = local_proto_2_local_clf_tmp_logit[:, 0]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 0] - min_logit_in_dim_0) / (
                                                                                             max_logit_in_dim_0 - min_logit_in_dim_0)

                        if max_logit_in_dim_1 == min_logit_in_dim_1:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = local_proto_2_local_clf_tmp_logit[:, 1]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 1] - min_logit_in_dim_1) / (
                                                                                             max_logit_in_dim_1 - min_logit_in_dim_1)
                    elif "IMG_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)
                    elif "Tabular_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)

                    if "SENT_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit[:, 1] - \
                                            normalized_local_proto_2_local_clf_tmp_logit[:, 0]
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "IMG_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "Tabular_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()

                    del normalized_local_proto_2_local_clf_tmp_logit
                    gc.collect()
                    # torch.cuda.empty_cache()

                    if torch.isnan(local_proto_2_local_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
                    else:
                        local_proto_2_local_clf_loss += criterion(
                            local_proto_2_local_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到局部分类器 的分类损失

                    global_model.to(device)
                    __, local_proto_2_global_clf_tmp_logit = global_model.only_clf_forward(local_proto_tensors)
                    global_model.cpu()
                    if torch.isnan(local_proto_2_global_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_global_clf_tmp_logit ###")
                    else:
                        local_proto_2_global_clf_loss += criterion(
                            local_proto_2_global_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到全局分类器 的分类损失
                    

                    # 群组决策差距（在本地增强群组公平性）
                    # 标签0和1 不同群组的预测分布差距
                    label_0_pred_distribution_gap, label_1_pred_distribution_gap = 0, 0
                    if "SENT_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "IMG_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "Tabular_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                    # 敏感属性 与 原型决策边界距离 的协方差

                    cov = get_cov_between_sensitive_attribute_and_prototype_decision_distance(
                        weight_list=local_proto_weight_list, z_list=local_proto_z_list,
                        prototype_decision_distance=local_proto_2_local_clf_decision_distance_list)
                    # print(f"cov: {cov}")
                    cov_abs = abs(cov)

                    lamda_list = [1, 1, 
                                  1, 1,
                                  1]  # FedPro思路
                    reg_list = [
                        local_proto_2_local_clf_loss, local_proto_2_global_clf_loss,
                        label_0_pred_distribution_gap, label_1_pred_distribution_gap,
                        cov_abs
                    ]
                    if float(batch_id) % 50 == 0:
                        # if iter_t != 0 and float(batch_id) % 10 == 0:
                        logger.info(f"### Origin task loss：{loss.item()} ;\n"
                                    f"local_proto_2_local_clf_loss：{round(local_proto_2_local_clf_loss, 5)} ;\n"
                                    f"local_proto_2_global_clf_loss: {round(local_proto_2_global_clf_loss, 5)} ;\n"

                                    f"label_0_pred_distribution_gap：{round(label_0_pred_distribution_gap, 5)} ;\n"
                                    f"label_1_pred_distribution_gap: {round(label_1_pred_distribution_gap, 5)} ;\n"

                                    f"cov_abs: {round(cov_abs, 5)} ;\n"

                                    f"in Batch_id:{batch_id} of Epoch:{epoch} in Client:{id}. ### ")
                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * reg_list[index]

                    # del sent_label_flag, sent_group_flag
                    # del client_i_group_1_label_1_feature_in_one_batch, client_i_group_0_label_1_feature_in_one_batch
                    # del client_i_group_1_label_0_feature_in_one_batch, client_i_group_0_label_0_feature_in_one_batch

                    loss.backward()
                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()
                        # 清空梯度
                        model.zero_grad()

                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels, batch_loss, loss
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels, batch_loss, loss

                    gc.collect()
                    # torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

                # logger.debug(f"GPU Memory :")
                # logger.debug(torch.cuda.memory_summary())
                # torch.cuda.empty_cache()
                # gc.collect()

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            # 记录GPU计算开始时间
            gpu_start_time = time.time()

            # 计算客户的 类原型
            # logger.info("~~~~~~~~~~~~~ 5. 计算客户的 类原型 ~~~~~~~~~~~~~~~")
            time_cost, result_dict = get_client_i_Prototype(param_dict, model, device, client_i_dataloader)
            # logger.info("~~~~~~~~~~~~~ 5. (用全局模型）计算客户的 类原型 ~~~~~~~~~~~~~~~")
            __, result_dict_from_global_model = get_client_i_Prototype(param_dict, global_model, device,
                                                                       client_i_dataloader)

            global_client_i_prototype_gap = 0
            with torch.no_grad():
                # Label 0, Group 0
                if result_dict['client_i_group_0_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_0_prototype = result_dict['client_i_group_0_label_0_prototype']

                    if result_dict_from_global_model['client_i_group_0_label_0_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_0_label_0_prototype']
                        g = result_dict_from_global_model['client_i_group_0_label_0_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_0_label_0_feature_list.append(client_i_group_0_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_0_prototype)

                # Label 0, Group 1
                if result_dict['client_i_group_1_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_0_prototype = result_dict['client_i_group_1_label_0_prototype']

                    if result_dict_from_global_model['client_i_group_1_label_0_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_1_label_0_prototype']
                        g = result_dict_from_global_model['client_i_group_1_label_0_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_1_label_0_feature_list.append(client_i_group_1_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_0_prototype)

                # Label 1, Group 0
                if result_dict['client_i_group_0_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_1_prototype = result_dict['client_i_group_0_label_1_prototype']

                    if result_dict_from_global_model['client_i_group_0_label_1_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_0_label_1_prototype']
                        g = result_dict_from_global_model['client_i_group_0_label_1_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_0_label_1_feature_list.append(client_i_group_0_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_1_prototype)

                # Label 1, Group 1
                if result_dict['client_i_group_1_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_1_prototype = result_dict['client_i_group_1_label_1_prototype']

                    if result_dict_from_global_model['client_i_group_1_label_1_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_1_label_1_prototype']
                        g = result_dict_from_global_model['client_i_group_1_label_1_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_1_label_1_feature_list.append(client_i_group_1_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_1_prototype)

            # del model
            # gc.collect()
            # # torch.cuda.empty_cache()
            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            # del model
            # gc.collect()
            # torch.cuda.empty_cache()

            prototype_gap_between_client_i_and_global_list.append(global_client_i_prototype_gap)
            # 注意这里的权重，不能用当前数据量占据整体数据的比例，要用参数聚合的权重（当前客户数据量占据所抽到的客户数据量总和的比例）
            # weighted_prototype_gap_between_client_i_and_global = (client_datasets_size_list[id] / training_dataset_size) * prototype_gap_between_client_i_and_global
            weighted_prototype_gap_between_client_i_and_global = client_i_aggregation_weight * global_client_i_prototype_gap
            weighted_prototype_gap_between_client_i_and_global_list.append(
                weighted_prototype_gap_between_client_i_and_global)

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
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_0_prototype_list) != 0:
                global_group_0_label_0_prototype_list.append(
                    EMA_frac * global_group_0_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_0_prototype
                )
            else:
                global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
        # Label 0, Group 1
        if len(weighted_global_group_1_label_0_feature_list) != 0:
            for proto in weighted_global_group_1_label_0_feature_list:
                global_group_1_label_0_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_0_prototype_list) != 0:
                global_group_1_label_0_prototype_list.append(
                    EMA_frac * global_group_1_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_0_prototype
                )
            else:
                global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
        # Label 1, Group 0
        if len(weighted_global_group_0_label_1_feature_list) != 0:
            for proto in weighted_global_group_0_label_1_feature_list:
                global_group_0_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_1_prototype_list) != 0:
                global_group_0_label_1_prototype_list.append(
                    EMA_frac * global_group_0_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_1_prototype
                )
            else:
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
        # Label 1, Group 1
        if len(weighted_global_group_1_label_1_feature_list) != 0:
            for proto in weighted_global_group_1_label_1_feature_list:
                global_group_1_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_1_prototype_list) != 0:
                global_group_1_label_1_prototype_list.append(
                    EMA_frac * global_group_1_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_1_prototype
                )
            else:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

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
            rep_theta_start_index, rep_theta_end_index = 0, len(get_parameters(rep_model))
            rep_theta_list.append(param[rep_theta_start_index: rep_theta_end_index])
            aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

            if prototype_gap_between_client_i_and_global_list[index] < prototype_gap_threshold:
                logger.info(
                    f"#@!#@!#@! Client {id}'s gap:{round(prototype_gap_between_client_i_and_global_list[index], 4)}, threshold:{round(prototype_gap_threshold, 4)} ;  ~ #@!#@!#@!")
                logger.info(
                    f"#@!#@!#@! The prototype_gap_between_client_{id}_and_global is too close, the Rep param not upload;  ~ #@!#@!#@!")
                counter_no_RepModelTrans += 1
                rep_aggregation_weights.append(0)
            else:
                logger.info(
                    f"#@!#@!#@! Client {id}'s gap:{round(prototype_gap_between_client_i_and_global_list[index], 4)}, threshold:{round(prototype_gap_threshold, 4)} ;  ~ #@!#@!#@!")

                counter_RepModelTrans += 1
                rep_aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

            # del selected_model
            # gc.collect()

        # 参数聚合
        try:
            if (len(aggregation_weights) != 0) and (sum(aggregation_weights) != 0):
                logger.info("Parameter aggregation")
                # 聚合完整的参数
                theta_list = np.array(theta_list, dtype=object)
                # FedAvg旧版论文的聚合权重是平均
                # theta_avg = np.mean(theta_list, 0).tolist()
                # FedAvg新版论文的聚合权重是数据占比
                # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
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
                    theta_avg[rep_theta_start_index: rep_theta_end_index] = rep_theta_list_avg

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)

                # del theta_list
                # gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

        logger.info(f"Communication Round {(iter_t + 1)}  Communication Cost: {accumulated_Communication_Cost} MB")

        # 更新全局原型Gap
        logger.info("Prototype Gap Threshold Aggregation")
        if len(weighted_prototype_gap_between_client_i_and_global_list) != 0:
            prototype_gap_threshold = np.array(weighted_prototype_gap_between_client_i_and_global_list).mean()

        logger.info("Testing before post training")
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                               testing_dataset_len)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader,
                                                                         testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

        # Server side post training
        logger.info("Server side post training")
        global_model.to(device)
        Server_side_post_training_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                                                learning_rate=param_dict['learning_rate'],
                                                                max_grad_norm=0)
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
            __, tmp_logit = global_model.only_clf_forward(x)
            if torch.isnan(tmp_logit).any():
                logger.info("### The tmp_logit is nan in Server side post training ###")
            else:
                post_training_loss = criterion(tmp_logit.to(device), tmp_label.to(device))
                post_training_loss = torch.sum(post_training_loss)
                post_training_loss.backward()
                Server_side_post_training_optimizer.step()

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
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

    logger.info("Training finish, save and return the global model.")
    logger.info(
        f"Rep Model Trans Time: {counter_RepModelTrans}, not trans time:{counter_no_RepModelTrans}, total:{counter_RepModelTrans + counter_no_RepModelTrans}")
    logger.info(
        f"Rep Model Trans Rate: {counter_RepModelTrans / (counter_RepModelTrans + counter_no_RepModelTrans)}, not trans Rate:{counter_no_RepModelTrans / (counter_RepModelTrans + counter_no_RepModelTrans)}")

    # Save global model
    # save_dir = f'./save_path/'
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"global_PDFFed.pt")
    # torch.save(global_model, save_path)

    total_communication_cost = accumulated_Communication_Cost
    return global_model, total_gpu_seconds, total_communication_cost


def PDF_Fed_V2_RepModel_DynamicTrans_Trainable_Constrain(device,
                                   global_model,
                                   algorithm_epoch_T, num_clients_K, communication_round_I,
                                   FL_fraction, FL_drop_rate,
                                   training_dataloaders,
                                   training_dataset,
                                   client_dataset_list,
                                   param_dict,
                                   testing_dataloader,
                                   testing_dataset_len):
    logger.info("!!!!!!!!!!!!   PDF_Fed_V2_RepModel_DynamicTrans_Trainable_Constrain   !!!!!!!!!!!!!!!!")
    counter_RepModelTrans = 0 # 记录传输了RepModel的次数
    counter_no_RepModelTrans = 0 # 记录无传输RepModel的次数

    accumulation_steps = int(256 / param_dict['batch_size'])

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    basic_path = param_dict['model_path']

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    local_Lagrangian_list = [torch.nn.Parameter(torch.tensor(1.), requires_grad=True) for _ in range(num_clients_K)]  # 预设多个拉格朗日乘子
    lambda_param_optimizer_list = [torch.optim.SGD([local_Lagrangian_list[i]], lr=0.1) for i in range(num_clients_K)]

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    num_of_class = 2
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

    if "SENT_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out.parameters()) * 4 / (1024 * 1024)

    elif "IMG_CLF" in param_dict["task"]:
        rep_model_MB_size = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_model_MB_size = sum(p.numel() for p in global_model.out_layer.parameters()) * 4 / (1024 * 1024)

    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # 自定义初始参数
    # try:
    #     EMA_frac = param_dict['EMA_frac']
    # except Exception:
    #     EMA_frac = 0.1

    EMA_frac = 0  # 相当于不使用EMA

    global_group_0_label_0_prototype_list = []
    global_group_1_label_0_prototype_list = []
    global_group_0_label_1_prototype_list = []
    global_group_1_label_1_prototype_list = []

    prototype_gap_threshold = -99999  # gap一开始很小。若本地的gap比全局的gap要小，证明局部的表征已经很贴近全局了，则不需要传表征参数

    accumulated_Communication_Cost = 0

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        users_gpu_seconds_list = [0] * num_clients_K

        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            # style="FedAvg",
            style="FedProx",
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
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            # 下发模型
            logger.info(f"Client {id} Init Local Model By Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            lambda_param = local_Lagrangian_list[id]

            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))

            lambda_param_optimizer = lambda_param_optimizer_list[id]

            client_i_dataloader = training_dataloaders[id]

            # Local Training
            logger.info("Start Local Training")
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

                    # 获取批内原型素材
                    # logger.info("#### 获取批内原型素材 #####")
                    with torch.no_grad():
                        try:
                            client_i_group_1_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_1_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).to(device)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_0_feature_in_one_batch = torch.stack(
                                [features[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).to(device)
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
                    # 局部原型 输入到局部分类器 的分类损失 # 局部原型 输入到全局分类器 的分类损失
                    local_proto_2_local_clf_loss, local_proto_2_global_clf_loss = 0, 0
                    # 获取原型驱动的分类任务素材
                    with torch.no_grad():
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
                            tmp_label = torch.zeros(1).to(device)
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
                    __, local_proto_2_local_clf_tmp_logit = model.only_clf_forward(local_proto_tensors)
                    if "SENT_CLF" in param_dict["task"]:
                        max_logit_in_dim_0 = torch.max(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        min_logit_in_dim_0 = torch.min(local_proto_2_local_clf_tmp_logit[:, 0], dim=0)[0].item()
                        max_logit_in_dim_1 = torch.max(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        min_logit_in_dim_1 = torch.min(local_proto_2_local_clf_tmp_logit[:, 1], dim=0)[0].item()
                        normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit.detach().clone()
                        if max_logit_in_dim_0 == min_logit_in_dim_0:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = local_proto_2_local_clf_tmp_logit[:, 0]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 0] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 0] - min_logit_in_dim_0) / (
                                                                                             max_logit_in_dim_0 - min_logit_in_dim_0)

                        if max_logit_in_dim_1 == min_logit_in_dim_1:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = local_proto_2_local_clf_tmp_logit[:, 1]
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit[:, 1] = (local_proto_2_local_clf_tmp_logit[
                                                                                      :, 1] - min_logit_in_dim_1) / (
                                                                                             max_logit_in_dim_1 - min_logit_in_dim_1)
                    elif "IMG_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)
                    elif "Tabular_CLF" in param_dict["task"]:
                        max_logit = torch.max(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        min_logit = torch.min(local_proto_2_local_clf_tmp_logit, dim=0)[0].item()
                        if max_logit == min_logit:
                            normalized_local_proto_2_local_clf_tmp_logit = local_proto_2_local_clf_tmp_logit
                        else:
                            normalized_local_proto_2_local_clf_tmp_logit = (
                                                                                       local_proto_2_local_clf_tmp_logit - min_logit) / (
                                                                                       max_logit - min_logit)

                    if "SENT_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit[:, 1] - \
                                            normalized_local_proto_2_local_clf_tmp_logit[:, 0]
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "IMG_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()
                    elif "Tabular_CLF" in param_dict["task"]:
                        decision_distance = normalized_local_proto_2_local_clf_tmp_logit.squeeze(1)
                        local_proto_2_local_clf_decision_distance_list = decision_distance.tolist()

                    del normalized_local_proto_2_local_clf_tmp_logit
                    gc.collect()
                    # torch.cuda.empty_cache()

                    if torch.isnan(local_proto_2_local_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_local_clf_tmp_logit ###")
                    else:
                        local_proto_2_local_clf_loss += criterion(
                            local_proto_2_local_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到局部分类器 的分类损失

                    global_model.to(device)
                    __, local_proto_2_global_clf_tmp_logit = global_model.only_clf_forward(local_proto_tensors)
                    global_model.cpu()
                    if torch.isnan(local_proto_2_global_clf_tmp_logit).any():
                        logger.info("### The tmp_logit is nan in local_proto_2_global_clf_tmp_logit ###")
                    else:
                        local_proto_2_global_clf_loss += criterion(
                            local_proto_2_global_clf_tmp_logit.to(device),
                            local_proto_2_global_clf_label_tensors.to(device)
                        ).mean().item()  # 局部原型 输入到全局分类器 的分类损失
                    

                    # 群组决策差距（在本地增强群组公平性）
                    # 标签0和1 不同群组的预测分布差距
                    label_0_pred_distribution_gap, label_1_pred_distribution_gap = 0, 0
                    if "SENT_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [logits[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "IMG_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).mean().to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                    elif "Tabular_CLF" in param_dict["task"]:
                        try:
                            client_i_group_1_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],
                                dim=0).mean().to(device)
                            client_i_group_0_label_0_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],
                                dim=0).mean().to(device)
                            label_0_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_0_pred_distribution_in_one_batch - client_i_group_0_label_0_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass
                        try:
                            client_i_group_1_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],
                                dim=0).to(device)
                            client_i_group_0_label_1_pred_distribution_in_one_batch = torch.stack(
                                [preds[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],
                                dim=0).to(device)
                            label_1_pred_distribution_gap += torch.norm(
                                client_i_group_1_label_1_pred_distribution_in_one_batch - client_i_group_0_label_1_pred_distribution_in_one_batch,
                                p=2).item()
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                    # 敏感属性 与 原型决策边界距离 的协方差

                    cov = get_cov_between_sensitive_attribute_and_prototype_decision_distance(
                        weight_list=local_proto_weight_list, z_list=local_proto_z_list,
                        prototype_decision_distance=local_proto_2_local_clf_decision_distance_list)
                    # print(f"cov: {cov}")
                    cov_abs = abs(cov)

                    clone_loss = loss.clone() * 0  # 正则项统计
                    # logger.info(f"### clone_loss：{clone_loss.item()}. ### ")

                    lamda_list = [1, 1, 
                                  1, 1,
                                  1]  # FedPro思路
                    reg_list = [
                        local_proto_2_local_clf_loss, local_proto_2_global_clf_loss,
                        label_0_pred_distribution_gap, label_1_pred_distribution_gap,
                        cov_abs
                    ]
                    if float(batch_id) % 50 == 0:
                        # if iter_t != 0 and float(batch_id) % 10 == 0:
                        logger.info(f"### Origin task loss：{loss.item()} ;\n"
                                    f"local_proto_2_local_clf_loss：{round(local_proto_2_local_clf_loss, 5)} ;\n"
                                    f"local_proto_2_global_clf_loss: {round(local_proto_2_global_clf_loss, 5)} ;\n"

                                    f"label_0_pred_distribution_gap：{round(label_0_pred_distribution_gap, 5)} ;\n"
                                    f"label_1_pred_distribution_gap: {round(label_1_pred_distribution_gap, 5)} ;\n"

                                    f"cov_abs: {round(cov_abs, 5)} ;\n"

                                    f"in Batch_id:{batch_id} of Epoch:{epoch} in Client:{id}. ### ")
                    for index, lamda in enumerate(lamda_list):
                        loss += lambda_param * lamda * reg_list[index]
                        clone_loss += lambda_param * reg_list[index]

                    loss.backward()
                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()

                        if float(batch_id) % 50 == 0:
                            logger.info(f'### lambda_param before update: {lambda_param.item()} ### ')
                        grad_lambda = torch.autograd.grad(
                            outputs=clone_loss,
                            inputs=lambda_param,
                            create_graph=False,
                            retain_graph=False,
                            only_inputs=True
                        )[0]
                        lambda_param.grad = -grad_lambda
                        lambda_param_optimizer.step()
                        if float(batch_id) % 50 == 0:
                            logger.info(f'### lambda_param after update: {lambda_param.item()} ### ')

                        local_Lagrangian_list[id] = lambda_param

                        # 清空梯度
                        model.zero_grad()

                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels, batch_loss, loss
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels, batch_loss, loss

                    # gc.collect()
                    # torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

                # logger.debug(f"GPU Memory :")
                # logger.debug(torch.cuda.memory_summary())
                # torch.cuda.empty_cache()
                # gc.collect()

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            # 记录GPU计算开始时间
            gpu_start_time = time.time()

            # 计算客户的 类原型
            # logger.info("~~~~~~~~~~~~~ 5. 计算客户的 类原型 ~~~~~~~~~~~~~~~")
            time_cost, result_dict = get_client_i_Prototype(param_dict, model, device, client_i_dataloader)
            # logger.info("~~~~~~~~~~~~~ 5. (用全局模型）计算客户的 类原型 ~~~~~~~~~~~~~~~")
            __, result_dict_from_global_model = get_client_i_Prototype(param_dict, global_model, device,
                                                                       client_i_dataloader)

            global_client_i_prototype_gap = 0
            with torch.no_grad():
                # Label 0, Group 0
                if result_dict['client_i_group_0_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_0_prototype = result_dict['client_i_group_0_label_0_prototype']

                    if result_dict_from_global_model['client_i_group_0_label_0_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_0_label_0_prototype']
                        g = result_dict_from_global_model['client_i_group_0_label_0_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_0_label_0_feature_list.append(client_i_group_0_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_0_prototype)

                # Label 0, Group 1
                if result_dict['client_i_group_1_label_0_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_0_prototype = result_dict['client_i_group_1_label_0_prototype']

                    if result_dict_from_global_model['client_i_group_1_label_0_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_1_label_0_prototype']
                        g = result_dict_from_global_model['client_i_group_1_label_0_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_1_label_0_feature_list.append(client_i_group_1_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_0_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_0_prototype)

                # Label 1, Group 0
                if result_dict['client_i_group_0_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_0_label_1_prototype = result_dict['client_i_group_0_label_1_prototype']

                    if result_dict_from_global_model['client_i_group_0_label_1_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_0_label_1_prototype']
                        g = result_dict_from_global_model['client_i_group_0_label_1_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_0_label_1_feature_list.append(client_i_group_0_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_0_label_1_prototype)

                # Label 1, Group 1
                if result_dict['client_i_group_1_label_1_prototype'] is not None:
                    # 得到客户的原型
                    client_i_group_1_label_1_prototype = result_dict['client_i_group_1_label_1_prototype']

                    if result_dict_from_global_model['client_i_group_1_label_1_prototype'] is not None:
                        w = 1 / 1
                        l = result_dict['client_i_group_1_label_1_prototype']
                        g = result_dict_from_global_model['client_i_group_1_label_1_prototype']
                        global_client_i_prototype_gap += w * torch.norm(l - g, p=2).item()

                    global_group_1_label_1_feature_list.append(client_i_group_1_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_1_feature_list.append(
                        client_i_aggregation_weight * client_i_group_1_label_1_prototype)

            # del model
            # gc.collect()
            # # torch.cuda.empty_cache()
            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            # del model
            # gc.collect()
            # torch.cuda.empty_cache()

            prototype_gap_between_client_i_and_global_list.append(global_client_i_prototype_gap)
            # 注意这里的权重，不能用当前数据量占据整体数据的比例，要用参数聚合的权重（当前客户数据量占据所抽到的客户数据量总和的比例）
            # weighted_prototype_gap_between_client_i_and_global = (client_datasets_size_list[id] / training_dataset_size) * prototype_gap_between_client_i_and_global
            weighted_prototype_gap_between_client_i_and_global = client_i_aggregation_weight * global_client_i_prototype_gap
            weighted_prototype_gap_between_client_i_and_global_list.append(weighted_prototype_gap_between_client_i_and_global)

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
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_0_prototype_list) != 0:
                global_group_0_label_0_prototype_list.append(
                    EMA_frac * global_group_0_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_0_prototype
                )
            else:
                global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
        # Label 0, Group 1
        if len(weighted_global_group_1_label_0_feature_list) != 0:
            for proto in weighted_global_group_1_label_0_feature_list:
                global_group_1_label_0_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_0_prototype_list) != 0:
                global_group_1_label_0_prototype_list.append(
                    EMA_frac * global_group_1_label_0_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_0_prototype
                )
            else:
                global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
        # Label 1, Group 0
        if len(weighted_global_group_0_label_1_feature_list) != 0:
            for proto in weighted_global_group_0_label_1_feature_list:
                global_group_0_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_0_label_1_prototype_list) != 0:
                global_group_0_label_1_prototype_list.append(
                    EMA_frac * global_group_0_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_0_label_1_prototype
                )
            else:
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
        # Label 1, Group 1
        if len(weighted_global_group_1_label_1_feature_list) != 0:
            for proto in weighted_global_group_1_label_1_feature_list:
                global_group_1_label_1_prototype += proto
            # 引入EMA式的全局Prototype更新
            if len(global_group_1_label_1_prototype_list) != 0:
                global_group_1_label_1_prototype_list.append(
                    EMA_frac * global_group_1_label_1_prototype_list[-1] + (
                                1 - EMA_frac) * global_group_1_label_1_prototype
                )
            else:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

        # 读取正常客户的参数
        theta_list = []
        rep_theta_list = []

        aggregation_weights = []
        rep_aggregation_weights = []

        # 获取参数聚合的素材
        for index, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path , weights_only=False)  # 持久化

            if "SENT_CLF" in param_dict["task"]:
                rep_model = selected_model.bert
            elif "IMG_CLF" in param_dict["task"]:
                rep_model = selected_model.shared_base
            elif "Tabular_CLF" in param_dict["task"]:
                rep_model = selected_model.shared_base

            param = get_parameters(selected_model)
            theta_list.append(param)
            rep_theta_start_index, rep_theta_end_index = 0, len(get_parameters(rep_model))
            rep_theta_list.append(param[rep_theta_start_index: rep_theta_end_index])
            aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

            if prototype_gap_between_client_i_and_global_list[index] < prototype_gap_threshold:
                logger.info(f"#@!#@!#@! Client {id}'s gap:{round(prototype_gap_between_client_i_and_global_list[index], 4)}, threshold:{round(prototype_gap_threshold, 4)} ;  ~ #@!#@!#@!")

                logger.info(f"#@!#@!#@! The prototype_gap_between_client_{id}_and_global is too close, the Rep param not upload;  ~ #@!#@!#@!")
                counter_no_RepModelTrans += 1
                rep_aggregation_weights.append(0)
            else:
                logger.info(f"#@!#@!#@! Client {id}'s gap:{round(prototype_gap_between_client_i_and_global_list[index], 4)}, threshold:{round(prototype_gap_threshold, 4)} ;  ~ #@!#@!#@!")

                counter_RepModelTrans += 1
                rep_aggregation_weights.append(client_datasets_size_list[id])  # 这个地方只需要读取客户的数据量，不用除以总量！

            # del selected_model
            # gc.collect()

        # 参数聚合
        try:
            if (len(aggregation_weights) != 0) and (sum(aggregation_weights) != 0):
                logger.info("Parameter aggregation")
                # 聚合完整的参数
                theta_list = np.array(theta_list, dtype=object)
                # FedAvg旧版论文的聚合权重是平均
                # theta_avg = np.mean(theta_list, 0).tolist()
                # FedAvg新版论文的聚合权重是数据占比
                # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
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
                    theta_avg[rep_theta_start_index: rep_theta_end_index] = rep_theta_list_avg

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)

                # del theta_list
                # gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

        logger.info(f"Communication Round {(iter_t + 1)}  Communication Cost: {accumulated_Communication_Cost} MB")

        # 更新全局原型Gap
        if len(weighted_prototype_gap_between_client_i_and_global_list) != 0:
            prototype_gap_threshold = np.array(weighted_prototype_gap_between_client_i_and_global_list).mean()

        logger.info("Testing before post training")
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                               testing_dataset_len)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader,
                                                                         testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
            FR = 1 - DEO
            HM = get_HM_by_two_value(accuracy, FR)
            logger.info(
                f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

        # Server side post training
        logger.info("Server side post training")
        global_model.to(device)
        Server_side_post_training_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                                                learning_rate=param_dict['learning_rate'],
                                                                max_grad_norm=0)
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
            __, tmp_logit = global_model.only_clf_forward(x)
            if torch.isnan(tmp_logit).any():
                logger.info("### The tmp_logit is nan in Server side post training ###")
            else:
                post_training_loss = criterion(tmp_logit.to(device), tmp_label.to(device))
                post_training_loss = torch.sum(post_training_loss)
                post_training_loss.backward()
                Server_side_post_training_optimizer.step()

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
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

    logger.info("Training finish, save and return the global model.")
    logger.info(f"Rep Model Trans Time: {counter_RepModelTrans}, not trans time:{counter_no_RepModelTrans}, total:{counter_RepModelTrans+counter_no_RepModelTrans}")
    logger.info(f"Rep Model Trans Rate: {counter_RepModelTrans/(counter_RepModelTrans+counter_no_RepModelTrans)}, not trans Rate:{counter_no_RepModelTrans/(counter_RepModelTrans+counter_no_RepModelTrans)}")

    # Save global model
    # save_dir = f'./save_path/'
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"global_PDFFed.pt")
    # torch.save(global_model, save_path)

    total_communication_cost = accumulated_Communication_Cost
    return global_model, total_gpu_seconds, total_communication_cost
