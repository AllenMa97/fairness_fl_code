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
from tool.utils import get_parameters, set_parameters, cos_sim, FL_fairness_and_accuracy_test_4_IMG_CLF, get_HM_by_two_value
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test
from hypothesis.generator import LatentGenerator, FigGenerator


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# PDFFed: Prototype-Driven Fair Federated Learning via Alignment, Selection, and Supervision
# FedAvg+FedProx的采样方法
# + 全局-局部原型驱动的表征空间对齐（在本地减缓数据异构导致的性能退化）
# + 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
# + 局部原型驱动的群组表征对比（在本地增强群组公平性）
# + 群组感知的全局原型分类损失差距（在本地增强群组公平性）
# + 群组感知的本地数据分类损失差距（在本地增强群组公平性）
# + 基于全局-局部原型表征差异的控制的选择上传机制（降低通信量，减缓系统异构的负面影响）
# + 基于全局原型与随机噪声的全局分类器监督（在全局减缓数据异构导致的性能退化）



def InfoNCE_loss(
        sample_feature: torch.Tensor,
        positive_feature: torch.Tensor,
        negative_feature_list: list,
        temperature: float = 0.07,
        device: str='cuda'
) -> torch.Tensor:
    """
    对比学习损失函数 (InfoNCE Loss)

    参数:
        sample_feature: 样本特征向量 [D]
        positive_feature: 正样本特征向量 [D]
        negative_feature_list: 负样本特征列表 [N][D]
        temperature: 温度系数 (默认0.07)

    返回:
        torch.Tensor: 计算得到的对比损失值
    """
    # 特征归一化
    sample = F.normalize(sample_feature, dim=0).to(device)
    positive = F.normalize(positive_feature, dim=0).to(device)
    negatives = torch.stack(negative_feature_list).to(device)
    negatives = F.normalize(negatives, dim=1).to(device)

    # 计算相似度
    pos_sim = torch.dot(sample, positive).to(device)  # 正样本相似度
    neg_sims = torch.matmul(sample, negatives.T).to(device)  # 负样本相似度矩阵 [1, N]

    # 合并logits
    logits = torch.cat(
        [
            torch.tensor([pos_sim / temperature]).to(device),
            neg_sims.view(-1) / temperature
        ]
    ).to(device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]).to(device)).item()
    return loss


def PDF_Fed(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):
    accumulation_steps = int(256 / param_dict['batch_size'])

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
    elif "IMG_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0

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

    # 自定义初始参数
    # try:
    #     EMA_frac = param_dict['EMA_frac']
    # except Exception:
    #     EMA_frac = 0.1

    EMA_frac = 0 # 相当于不使用EMA

    global_group_0_label_0_prototype_list = []
    global_group_1_label_0_prototype_list = []
    global_group_0_label_1_prototype_list = []
    global_group_1_label_1_prototype_list = []

    prototype_gap_threshold = -99999 # gap一开始很小。若本地的gap比全局的gap要小，证明局部的表征已经很贴近全局了，则不需要传表征参数

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

            client_i_group_0_label_0_feature_list = []
            client_i_group_1_label_0_feature_list = []

            client_i_group_0_label_1_feature_list = []
            client_i_group_1_label_1_feature_list = []

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
                    # labels尺寸 [batch_size]
                    labels = batch["labels"].to(device)
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
                        batch_loss = criterion(activated_preds, labels)

                    elif "IMG_CLF" in param_dict["task"]:
                        # preds尺寸 [batch_size, 1]
                        # features尺寸 [batch_size, emb_dim]
                        preds, features = model(imgs)
                        batch_loss = criterion(preds[:, 0], labels.float())

                    loss = torch.sum(batch_loss) / true_batch_size


                    # 基于分类的两个损失global_GroupProto4CLFLoss 和 local_GroupProto4CLFLoss，启发于
                    # (AAAI 2025) FedSA: A Unifed Representation Learning via Semantic Anchors for Prototype-based Federated Learning
                    # https://arxiv.org/pdf/2501.05496
                    # 这篇文章提到3个Observations：
                    # 1 数据分布的偏移(Data Distribution Drift) 会 导致表征不一致(Representation inconsistency for the same input)，即我理解的表征空间的偏移(Drift in Representation Space)
                    # 2 从而进一步导致倾斜的原型对齐(Skewed prototype alignment) 以及 分类器分离(Classifier divergence)，
                    # 原型对齐(Skewed prototype alignment)意味着全局级别不同类别的原型之间的区分度降低
                    # 分类器分离(Classifier divergence)意味着不同客户的分类边界将受到影响
                    #
                    # Prototype的引入以及FedProto的成功已经证实这种方案可以在一定程度上缓解了表征空间的偏移的问题了
                    # 对于减缓分类器的分离，我们借鉴了这篇文章里面的Anchor-based classifer calibration的思路，同时也是考虑了FedProto中没有针对clf进行优化的不足，我们引入了 global_GroupProto4CLFLoss 和 local_GroupProto4CLFLoss
                    #
                    with torch.no_grad():
                        # 添加原型素材
                        sent_label_flag = labels.gt(0.5).float().reshape([-1, 1]).cpu()
                        sent_group_flag = protecteds.gt(0.5).float().reshape([-1, 1]).cpu()

                        client_i_group_1_label_1_flag = (sent_group_flag * sent_label_flag)[:,0].bool().tolist()
                        client_i_group_0_label_1_flag = (( 1 - sent_group_flag) * sent_label_flag )[:,0].bool().tolist()
                        client_i_group_1_label_0_flag = (sent_group_flag * (1 - sent_label_flag))[:,0].bool().tolist()
                        client_i_group_0_label_0_flag = ((1 - sent_group_flag) * (1 - sent_label_flag))[:,0].bool().tolist()

                        try:
                            client_i_group_1_label_1_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_1_label_1_flag) if item],dim=0).to(device)
                            client_i_group_1_label_1_feature_list.append(client_i_group_1_label_1_feature_in_one_batch)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_1_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_0_label_1_flag) if item],dim=0).to(device)
                            client_i_group_0_label_1_feature_list.append(client_i_group_0_label_1_feature_in_one_batch)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_1_label_0_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_1_label_0_flag) if item],dim=0).to(device)
                            client_i_group_1_label_0_feature_list.append(client_i_group_1_label_0_feature_in_one_batch)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        try:
                            client_i_group_0_label_0_feature_in_one_batch = torch.stack([features[index] for index, item in enumerate(client_i_group_0_label_0_flag) if item],dim=0).to(device)
                            client_i_group_0_label_0_feature_list.append(client_i_group_0_label_0_feature_in_one_batch)
                        except Exception:
                            # 有异常则表示batch内没抽到这个group&这个label的数据
                            pass

                        # 计算Proto gap和 Proto Alignment gap
                        # 每种原型的局部与全局差异
                        (group_0_label_0_feature_gap, group_0_label_1_feature_gap) = 0, 0
                        (group_1_label_0_feature_gap, group_1_label_1_feature_gap) = 0, 0
                        # 每种局部原型的群组差异
                        # (label_0_feature_gap, label_1_feature_gap) = 0, 0
                        # 每种全局原型的分类损失
                        (global_group_0_label_0_clf_loss, global_group_0_label_1_clf_loss) = 0, 0
                        (global_group_1_label_0_clf_loss, global_group_1_label_1_clf_loss) = 0, 0
                        # 每种局部原型的分类损失
                        (local_group_0_label_0_clf_loss, local_group_0_label_1_clf_loss) = 0, 0
                        (local_group_1_label_0_clf_loss, local_group_1_label_1_clf_loss) = 0, 0


                        local_one_batch_proto_list = []

                        # Label 0, Group 0
                        if len(global_group_0_label_0_prototype_list) != 0:
                            if "SENT_CLF" in param_dict["task"]:
                                tmp_label = torch.tensor([1,0]).float().to(device)
                            elif "IMG_CLF" in param_dict["task"]:
                                tmp_label = torch.zeros(1).to(device)
                            g = global_group_0_label_0_prototype_list[-1].to(device)
                            accumulated_Communication_Cost += prototype_MB_size  # 下载一个Proto所带来的通信消耗

                            # 全局原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                            __, tmp_logit = model.only_clf_forward(g)
                            global_group_0_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label).item()

                            try:
                                client_i_group_0_label_0_one_batch_proto = client_i_group_0_label_0_feature_in_one_batch.mean(dim=0)
                                l = client_i_group_0_label_0_one_batch_proto.to(device)
                                local_one_batch_proto_list.append(l)

                                # 全局-本地类原型的差异（欧几里得距离） -- 全局-局部原型驱动的表征空间对齐（在本地减缓数据异构导致的性能退化）
                                gap = torch.norm(g-l, p=2).item()
                                # label_0_feature_gap += gap
                                # group_0_label_0_feature_gap = 0.5 * float(gap ** 2)
                                group_0_label_0_feature_gap = gap


                                # 局部原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_0_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label).item()
                            except Exception:
                                # 有异常则表示batch内没抽到这个group&这个label的数据
                                pass

                        # Label 0, Group 1
                        if len(global_group_1_label_0_prototype_list) != 0:
                            if "SENT_CLF" in param_dict["task"]:
                                tmp_label = torch.tensor([1, 0]).float().to(device)
                            elif "IMG_CLF" in param_dict["task"]:
                                tmp_label = torch.zeros(1).to(device)
                            g = global_group_1_label_0_prototype_list[-1].to(device)
                            accumulated_Communication_Cost += prototype_MB_size  # 下载一个Proto所带来的通信消耗

                            # 全局原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                            __, tmp_logit = model.only_clf_forward(g)
                            global_group_1_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label).item()

                            try:
                                client_i_group_1_label_0_one_batch_proto = client_i_group_1_label_0_feature_in_one_batch.mean(dim=0)
                                l = client_i_group_1_label_0_one_batch_proto.to(device)
                                local_one_batch_proto_list.append(l)


                                # 全局-本地类原型的差异（欧几里得距离） -- 全局-局部原型驱动的表征空间对齐（在本地减缓数据异构导致的性能退化）
                                gap = torch.norm(g-l, p=2).item()
                                # label_0_feature_gap += gap
                                # group_1_label_0_feature_gap = 0.5 * float(gap ** 2)
                                group_1_label_0_feature_gap = gap


                                # 局部原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_1_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label).item()
                            except Exception:
                                # 有异常则表示batch内没抽到这个group&这个label的数据
                                pass

                        # Label 1, Group 0
                        if len(global_group_0_label_1_prototype_list) != 0:
                            if "SENT_CLF" in param_dict["task"]:
                                tmp_label = torch.tensor([0, 1]).float().to(device)
                            elif "IMG_CLF" in param_dict["task"]:
                                tmp_label = torch.ones(1).to(device)
                            g = global_group_0_label_1_prototype_list[-1].to(device)
                            accumulated_Communication_Cost += prototype_MB_size  # 下载一个Proto所带来的通信消耗

                            # 全局原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                            __, tmp_logit = model.only_clf_forward(g)
                            global_group_0_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).item()

                            try:
                                client_i_group_0_label_1_one_batch_proto = client_i_group_0_label_1_feature_in_one_batch.mean(dim=0)
                                l = client_i_group_0_label_1_one_batch_proto.to(device)
                                local_one_batch_proto_list.append(l)


                                # 全局-本地类原型的差异（欧几里得距离） -- 全局-局部原型驱动的表征空间对齐（在本地减缓数据异构导致的性能退化）
                                gap = torch.norm(g-l, p=2).item()
                                # label_1_feature_gap += gap
                                # group_0_label_1_feature_gap = 0.5 * float(gap ** 2)
                                group_0_label_1_feature_gap = gap


                                # 局部原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_0_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).item()
                            except Exception:
                                # 有异常则表示batch内没抽到这个group&这个label的数据
                                pass

                        # Label 1, Group 1
                        if len(global_group_1_label_1_prototype_list) != 0:
                            if "SENT_CLF" in param_dict["task"]:
                                tmp_label = torch.tensor([0, 1]).float().to(device)
                            elif "IMG_CLF" in param_dict["task"]:
                                tmp_label = torch.ones(1).to(device)
                            g = global_group_1_label_1_prototype_list[-1].to(device)
                            accumulated_Communication_Cost += prototype_MB_size  # 下载一个Proto所带来的通信消耗

                            # 全局原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                            __, tmp_logit = model.only_clf_forward(g)
                            global_group_1_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).item()

                            try:
                                client_i_group_1_label_1_one_batch_proto = client_i_group_1_label_1_feature_in_one_batch.mean(dim=0)
                                l = client_i_group_1_label_1_one_batch_proto.to(device)
                                local_one_batch_proto_list.append(l)


                                # 全局-本地类原型的差异（欧几里得距离） -- 全局-局部原型驱动的表征空间对齐（在本地减缓数据异构导致的性能退化）
                                gap = torch.norm(g-l, p=2).item()
                                # label_1_feature_gap += gap
                                # group_1_label_1_feature_gap = 0.5 * float(gap ** 2)
                                group_1_label_1_feature_gap = gap



                                # 局部原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_1_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).item()
                            except Exception:
                                # 有异常则表示batch内没抽到这个group&这个label的数据
                                pass

                        # 全局-局部原型驱动的群组表征对比（在本地增强群组公平性）
                        if len(local_one_batch_proto_list) == 4:
                            # 长度为4证明这个batch内两个群组两个类别的样本都有了，可以进行对比
                            # Label 0, Group 0
                            contrastive_loss_1 = InfoNCE_loss(sample_feature=local_one_batch_proto_list[0].to(device),
                                                              positive_feature=local_one_batch_proto_list[1].to(device),
                                                              negative_feature_list=
                                                                  [
                                                                        local_one_batch_proto_list[2],
                                                                        local_one_batch_proto_list[3],
                                                                        global_group_0_label_1_prototype_list[-1].to(device),
                                                                        global_group_1_label_1_prototype_list[-1].to(device),
                                                                  ]
                                                              )
                            # Label 0, Group 1
                            # contrastive_loss_2 = InfoNCE_loss(sample_feature=local_one_batch_proto_list[1].to(device),
                            #                                   positive_feature=global_group_1_label_0_prototype_list[-1].to(device),
                            #                                   negative_feature_list=
                            #                                       [
                            #                                           local_one_batch_proto_list[2],
                            #                                           local_one_batch_proto_list[3],
                            #                                           global_group_0_label_1_prototype_list[-1].to(device),
                            #                                           global_group_1_label_1_prototype_list[-1].to(device),
                            #                                       ]
                            #                                   )

                            # Label 1, Group 0
                            contrastive_loss_3 = InfoNCE_loss(sample_feature=local_one_batch_proto_list[2].to(device),
                                                              positive_feature=local_one_batch_proto_list[3].to(device),
                                                              negative_feature_list=
                                                                  [
                                                                      local_one_batch_proto_list[0],
                                                                      local_one_batch_proto_list[1],
                                                                      global_group_0_label_0_prototype_list[-1].to(device),
                                                                      global_group_1_label_0_prototype_list[-1].to(device),
                                                                  ]
                                                              )
                            # Label 1, Group 1
                            # contrastive_loss_4 = InfoNCE_loss(sample_feature=local_one_batch_proto_list[3].to(device),
                            #                                   positive_feature=global_group_1_label_1_prototype_list[-1].to(device),
                            #                                   negative_feature_list=
                            #                                       [
                            #                                           local_one_batch_proto_list[0],
                            #                                           local_one_batch_proto_list[1],
                            #                                           global_group_0_label_0_prototype_list[-1].to(device),
                            #                                           global_group_1_label_0_prototype_list[-1].to(device),
                            #                                       ]
                            #                                   )

                            # local_rep_global_contrastive_loss = (contrastive_loss_1 + contrastive_loss_2 + contrastive_loss_3 + contrastive_loss_4)/4
                            local_rep_global_contrastive_loss = (contrastive_loss_1  + contrastive_loss_3 ) / 2

                            loss += local_rep_global_contrastive_loss

                            if float(batch_id) % 50 == 0:
                                logger.info(f"local_rep_global_contrastive_loss: {local_rep_global_contrastive_loss} in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")

                        # 群组感知的全局原型分类损失差距（在本地增强群组公平性）
                        global_group_0_clf_loss = global_group_0_label_0_clf_loss + global_group_0_label_1_clf_loss
                        global_group_1_clf_loss = global_group_1_label_0_clf_loss + global_group_1_label_1_clf_loss
                        if (global_group_0_clf_loss != 0) and (global_group_1_clf_loss != 0):
                            global_group_loss_gap = abs(global_group_0_clf_loss - global_group_1_clf_loss)
                            if float(batch_id) % 50 == 0:
                                logger.info(f"global_group_loss_gap: {global_group_loss_gap} in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")
                            loss += global_group_loss_gap

                        # 群组感知的本地数据分类损失差距（在本地增强群组公平性）
                        '''
                        这个地方可以形式化为我们用Lagrangian approach来构建了一个Constrain
                        具体的数学写法可以参考AAAI 2024的文章 https://arxiv.org/pdf/2312.05551v1 里面的Eq.10到Eq.12
                        考虑到我们的约束项前面乘的超参数并没有进行更新，所以可以不写成所谓的拉格朗日优化形式
                        不过别人的Eq.10-Eq.12是用的Fairness Matrix的gap，我们用的是Performance的gap，所以说我们的gap并不直接
                        '''
                        group_flag = protecteds.gt(0.5)
                        one_batch_group_1_count = sum(group_flag)
                        one_batch_group_0_count = true_batch_size - sum(group_flag)
                        if (one_batch_group_1_count != 0) and (one_batch_group_0_count != 0):
                            one_batch_group_1_avg_loss = float(
                                sum(batch_loss[group_flag]) / one_batch_group_1_count)
                            one_batch_group_0_avg_loss = float(
                                (sum(batch_loss) - sum(batch_loss[group_flag])) / one_batch_group_0_count)
                            one_batch_group_avg_loss_gap = abs(
                                one_batch_group_0_avg_loss - one_batch_group_1_avg_loss)
                            if float(batch_id) % 50 == 0:
                                logger.info(f"one_batch_group_avg_loss_gap: {one_batch_group_avg_loss_gap} "
                                            f"in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")
                            loss += one_batch_group_avg_loss_gap

                        lamda_list = [1, 1, 1, 1,
                                      1, 1, 1, 1,
                                      1, 1, 1, 1,]  # FedPro思路
                        reg_list = [group_0_label_0_feature_gap, group_0_label_1_feature_gap, # 全局-本地类原型的差异（欧几里得距离） -- 全局-局部原型驱动的表征空间对齐（在本地减缓数据异构导致的性能退化）
                                    group_1_label_0_feature_gap, group_1_label_1_feature_gap, # 全局-本地类原型的差异（欧几里得距离） -- 全局-局部原型驱动的表征空间对齐（在本地减缓数据异构导致的性能退化）
                                    global_group_0_label_0_clf_loss, global_group_0_label_1_clf_loss, # 全局原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                    global_group_1_label_0_clf_loss, global_group_1_label_1_clf_loss, # 全局原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                    local_group_0_label_0_clf_loss, local_group_0_label_1_clf_loss, # 局部原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                    local_group_1_label_0_clf_loss, local_group_1_label_1_clf_loss, # 局部原型的训练损失 -- 全局-局部原型驱动的局部分类器监督（在本地减缓数据异构导致的性能退化）
                                    ]
                        if iter_t != 0 and float(batch_id) % 10 == 0:
                            logger.info(f"Origin task loss：{loss.item()} ;\n"
                                        f"group_0_label_0_feature_gap：{group_0_label_0_feature_gap} ;\n"
                                        f"global_group_0_label_0_clf_loss: {global_group_0_label_0_clf_loss} ;\n"
                                        f"local_group_0_label_0_clf_loss: {local_group_0_label_0_clf_loss} ;\n"
                                        
                                        f"group_1_label_0_feature_gap: {group_1_label_0_feature_gap} ;\n"
                                        f"global_group_1_label_0_clf_loss: {global_group_1_label_0_clf_loss} ;\n"
                                        f"local_group_1_label_0_clf_loss: {local_group_1_label_0_clf_loss} ;\n"
                                        
                                        f"group_0_label_1_feature_gap: {group_0_label_1_feature_gap} ;\n"
                                        f"global_group_0_label_1_clf_loss: {global_group_0_label_1_clf_loss} ;\n"
                                        f"local_group_0_label_1_clf_loss: {local_group_0_label_1_clf_loss} ;\n"
                                        
                                        f"group_1_label_1_feature_gap: {group_1_label_1_feature_gap} ;\n"
                                        f"global_group_1_label_1_clf_loss: {global_group_1_label_1_clf_loss} ;\n"
                                        f"local_group_1_label_1_clf_loss: {local_group_1_label_1_clf_loss} ;\n"
                                        
                                        f"in Batch_id:{batch_id} of Epoch:{epoch} in Client:{id}.")
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
                    # break # 这里记得删除哦

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

            # prototype_gap_between_client_i_and_global = 0
            tmp_list = []
            # 计算客户的 类原型
            with torch.no_grad():
                group_0_label_0_flag = (len(client_i_group_0_label_0_feature_list) != 0)
                group_1_label_0_flag = (len(client_i_group_1_label_0_feature_list) != 0)
                group_0_label_1_flag = (len(client_i_group_0_label_1_feature_list) != 0)
                group_1_label_1_flag = (len(client_i_group_1_label_1_feature_list) != 0)


                # Label 0, Group 0
                if group_0_label_0_flag:
                    # 得到客户的原型
                    # client_i_group_0_label_0_prototype = torch.stack(client_i_group_0_label_0_feature_list, dim=0).mean(dim=0)
                    client_i_group_0_label_0_prototype = torch.concatenate(client_i_group_0_label_0_feature_list, dim=0).mean(dim=0)
                    accumulated_Communication_Cost += prototype_MB_size  # 上传一个Proto所带来的通信消耗

                    # 基于全局-局部原型表征差异的控制的选择上传机制（降低通信量，减缓系统异构的负面影响）
                    if len(global_group_0_label_0_prototype_list) != 0:
                        group_0_label_0_prototype_gap_between_client_i_and_global = torch.norm(client_i_group_0_label_0_prototype - global_group_0_label_0_prototype_list[-1], p=2).item()
                        tmp_list.append(group_0_label_0_prototype_gap_between_client_i_and_global)

                    global_group_0_label_0_feature_list.append(client_i_group_0_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_0_feature_list.append(client_i_aggregation_weight * client_i_group_0_label_0_prototype)

                # Label 0, Group 1
                if group_1_label_0_flag:
                    # 得到客户的原型
                   # client_i_group_1_label_0_prototype = torch.stack(client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)
                    client_i_group_1_label_0_prototype = torch.concatenate(client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)
                    accumulated_Communication_Cost += prototype_MB_size  # 上传一个Proto所带来的通信消耗

                    # 基于全局-局部原型表征差异的控制的选择上传机制（降低通信量，减缓系统异构的负面影响）
                    if len(global_group_1_label_0_prototype_list) != 0:
                        group_1_label_0_prototype_gap_between_client_i_and_global = torch.norm(client_i_group_1_label_0_prototype - global_group_1_label_0_prototype_list[-1], p=2).item()
                        tmp_list.append(group_1_label_0_prototype_gap_between_client_i_and_global)

                    global_group_1_label_0_feature_list.append(client_i_group_1_label_0_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_0_feature_list.append(client_i_aggregation_weight * client_i_group_1_label_0_prototype)

                # Label 1, Group 0
                if group_0_label_1_flag:
                    # 得到客户的原型
                    # client_i_group_0_label_1_prototype = torch.stack(client_i_group_0_label_1_feature_list, dim=0).mean(dim=0)
                    client_i_group_0_label_1_prototype = torch.concatenate(client_i_group_0_label_1_feature_list, dim=0).mean(dim=0)
                    accumulated_Communication_Cost += prototype_MB_size  # 上传一个Proto所带来的通信消耗

                    # 基于全局-局部原型表征差异的控制的选择上传机制（降低通信量，减缓系统异构的负面影响）
                    if len(global_group_0_label_1_prototype_list) != 0:
                        group_0_label_1_prototype_gap_between_client_i_and_global = torch.norm(client_i_group_0_label_1_prototype - global_group_0_label_1_prototype_list[-1], p=2).item()
                        tmp_list.append(group_0_label_1_prototype_gap_between_client_i_and_global)

                    global_group_0_label_1_feature_list.append(client_i_group_0_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_0_label_1_feature_list.append(client_i_aggregation_weight * client_i_group_0_label_1_prototype)

                # Label 1, Group 1
                if group_1_label_1_flag:
                    # 得到客户的原型
                    # client_i_group_1_label_1_prototype = torch.stack(client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
                    client_i_group_1_label_1_prototype = torch.concatenate(client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
                    accumulated_Communication_Cost += prototype_MB_size  # 上传一个Proto所带来的通信消耗

                    # 基于全局-局部原型表征差异的控制的选择上传机制（降低通信量，减缓系统异构的负面影响）
                    if len(global_group_1_label_1_prototype_list) != 0:
                        group_1_label_1_prototype_gap_between_client_i_and_global = torch.norm(client_i_group_1_label_1_prototype - global_group_1_label_1_prototype_list[-1], p=2).item()
                        tmp_list.append(group_1_label_1_prototype_gap_between_client_i_and_global)

                    global_group_1_label_1_feature_list.append(client_i_group_1_label_1_prototype)
                    # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                    weighted_global_group_1_label_1_feature_list.append(client_i_aggregation_weight * client_i_group_1_label_1_prototype)


            if len(tmp_list) != 0:
                prototype_gap_between_client_i_and_global = sum(tmp_list) / len(tmp_list)

                # 注意这里的权重，不能用当前数据量占据整体数据的比例，要用参数聚合的权重（当前客户数据量占据所抽到的客户数据量总和的比例）
                # weighted_prototype_gap_between_client_i_and_global = (client_datasets_size_list[id] / training_dataset_size) * prototype_gap_between_client_i_and_global
                weighted_prototype_gap_between_client_i_and_global = client_i_aggregation_weight * prototype_gap_between_client_i_and_global
                prototype_gap_between_client_i_and_global_list.append(prototype_gap_between_client_i_and_global)
                weighted_prototype_gap_between_client_i_and_global_list.append(weighted_prototype_gap_between_client_i_and_global)


            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            del model
            gc.collect()
            torch.cuda.empty_cache()

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
                    EMA_frac * global_group_0_label_0_prototype_list[-1] + (1-EMA_frac) * global_group_0_label_0_prototype
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
                    EMA_frac * global_group_1_label_0_prototype_list[-1] + (1-EMA_frac) * global_group_1_label_0_prototype
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
                    EMA_frac * global_group_0_label_1_prototype_list[-1] + (1-EMA_frac) * global_group_0_label_1_prototype
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
                    EMA_frac * global_group_1_label_1_prototype_list[-1] + (1 - EMA_frac) * global_group_1_label_1_prototype
                )
            else:
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

        # 读取正常客户的参数
        theta_list = []
        rep_theta_start_index, rep_theta_end_index = 0, 0
        rep_theta_list = []

        aggregation_weights = []
        rep_aggregation_weights = []

        # 记录每个客户的prototype有没有跟全局的Prototype产生较大的变化
        # 如果局部的Prototype和全局的Prototype差异很小，就不上传Rep部分的参数了
        logger.info(f"prototype_gap_between_client_i_and_global_list: {prototype_gap_between_client_i_and_global_list}")
        flag_list = [item > prototype_gap_threshold for item in prototype_gap_between_client_i_and_global_list]

        # 获取参数聚合的素材
        for index, id in enumerate(idxs_users):
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path)  # 持久化

            if "SENT_CLF" in param_dict["task"]:
                rep_model = selected_model.bert

            elif "IMG_CLF" in param_dict["task"]:
                rep_model = selected_model.shared_base

            param = get_parameters(selected_model)
            theta_list.append(param)
            rep_theta_start_index = 0
            rep_theta_end_index = len(get_parameters(rep_model))
            rep_theta_list.append(param[rep_theta_start_index : rep_theta_end_index])
            aggregation_weights.append(client_datasets_size_list[id]) # 这个地方只需要读取客户的数据量，不用除以总量！

            # 无论如何CLF部分的参数都是要上传的，先记录
            accumulated_Communication_Cost += clf_model_MB_size  # 上传一个CLF所带来的通信消耗

            if len(flag_list) != 0:
                if flag_list[index]:
                    tmp = 1 # 若gap大于阈值，证明变化很大，需要传Rep部分的参数
                    accumulated_Communication_Cost += rep_model_MB_size  # 上传一个rep所带来的通信消耗
                else:
                    tmp = 0 # 若gap小于阈值，证明变化很小，不需要传Rep部分的参数
                    logger.info(f"The prototype gap of Client {id} is narrow, do not need to upload the representation model!!")
            else:
                # flag_list为空，则证明是第一轮，需要传Rep部分的参数
                tmp = 1
                accumulated_Communication_Cost += rep_model_MB_size  # 上传一个rep所带来的通信消耗


            rep_aggregation_weights.append(tmp * client_datasets_size_list[id]) # 这个地方只需要读取客户的数据量，不用除以总量！

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
                    theta_avg[rep_theta_start_index : rep_theta_end_index] = rep_theta_list_avg

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)

                del theta_list
                gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

        logger.info(f"Communication Round {(iter_t + 1)}  Communication Cost: {accumulated_Communication_Cost} MB")


        # 更新全局原型Gap
        if len(weighted_prototype_gap_between_client_i_and_global_list) != 0:
            prototype_gap_threshold = np.array(weighted_prototype_gap_between_client_i_and_global_list).mean()

        logger.info(f"Communication Round {(iter_t + 1)}  prototype_gap_threshold: {prototype_gap_threshold}")


        logger.info("Testing before post training")
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


        # Server side post training
        logger.info("Server side post training")
        global_model.to(device)
        Server_side_post_training_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        if "SENT_CLF" in param_dict["task"]:
            Server_side_post_training_optimizer.set_parameters(list(global_model.out.named_parameters()))
        elif "IMG_CLF" in param_dict["task"]:
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
            else:
                if "SENT_CLF" in param_dict["task"]:
                    tmp_label = torch.tensor([1, 0]).float().to(device)
                elif "IMG_CLF" in param_dict["task"]:
                    tmp_label = torch.zeros(1).to(device)

            __, tmp_logit = global_model.only_clf_forward(x)

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


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_PDFFed.pt")
    torch.save(global_model, save_path)

    total_communication_cost = accumulated_Communication_Cost
    return global_model, total_gpu_seconds, total_communication_cost
