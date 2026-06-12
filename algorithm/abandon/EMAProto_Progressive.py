import copy
import os
import gc
import random
import time
import torch
import math
import numpy as np
import torch.autograd as autograd

from tool.logger import *
from tool.utils import get_parameters, set_parameters, cos_sim
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test
from hypothesis.generator import LatentGenerator

torch.autograd.set_detect_anomaly(True)

def InfoNCE_loss_of_samples(z, pos_indexs, neg_indexs, tau=0.5):
    """
    InfoNCE损失函数
    :param z: 模型的输出嵌入（通常是正样本的嵌入）
    :param pos_indexs: 正样本的索引
    :param neg_indexs: 负样本的索引
    :param tau: 温度参数，用于调整相似度的尺度
    :return: InfoNCE损失
    """
    z_len = len(z)
    try:
        InfoNCE_loss_avg, InfoNCE_loss_sum = 0, 0
        for q_index, query_feature in enumerate(z):
            one_query_InfoNCE_loss = 0
            pos_len = len(pos_indexs)
            if pos_len != 0:
                for p_index in pos_indexs:
                    if q_index == p_index:
                        continue
                    else:
                        # 计算样本和一个正样本的余弦相似度
                        pos_feature = z[p_index]
                        pos_sim = torch.exp(query_feature.dot(pos_feature) / tau)
                    neg_sim = 0
                    for n_index in neg_indexs:
                        if q_index == n_index:
                            continue
                        else:
                            # 计算样本和对应的所有负样本的余弦相似度
                            neg_feature = z[n_index]
                            neg_sim += torch.exp(query_feature.dot(neg_feature) / tau)
                    # 计算单个样本归一化的对数和指数
                    one_query_InfoNCE_loss += torch.exp(pos_sim/(pos_sim+neg_sim)) / pos_len

            InfoNCE_loss_sum += one_query_InfoNCE_loss
            InfoNCE_loss_avg += one_query_InfoNCE_loss / z_len

        return InfoNCE_loss_avg, InfoNCE_loss_sum
    except Exception:
        return 0, 0

# 参考DOSFL
def DOSFL_DISTILLDATA(param_dict, model, client_i_dataloader, device):
    # 自设参数
    Sd = param_dict['batch_size']
    # Sd = 256
    E = param_dict['algorithm_epoch_T']
    # DOSFL超参数
    emb_dim = 768
    Ed = 10
    η0 = 0.01
    alpha = 0.1
    τ = 10
    Generator = LatentGenerator(emb_dim).to(device)
    for param in Generator.parameters():
        param.requires_grad = False
    noise_inputs_embeds = torch.rand([Sd, param_dict['max_len'], emb_dim], device=device)
    distilled_samples = Generator(noise_inputs_embeds).to(device)
    distilled_samples.requires_grad = True
    distilled_samples.retain_grad()

    noise_attention_mask = torch.tensor(
        [[1 for i in range(param_dict['max_len'])] for j in range(Sd)], device=device)
    noise_token_type_ids = torch.tensor(
        [[0 for i in range(param_dict['max_len'])] for j in range(Sd)], device=device)
    distilled_labels = torch.round(torch.rand(Sd, device=device)).long()
    # 初始化distilled_learning_rate
    distilled_learning_rate = 0 * torch.randn(1) + η0
    distilled_learning_rate.requires_grad = True


    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method="DOSFL_ADAM", learning_rate=distilled_learning_rate.item(), decay_steps=τ, max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))

    for e in range(0, E):
        # distilled_learning_rate要每个epoch更新，所以要多次设置
        optimizer._set_rate(learning_rate=distilled_learning_rate.item())
        model.train()

        # 原论文 Algorithm 1 Line 16-23，更新模型参数
        for i in range(0, Ed):
            for j in range(0, Sd):
                # features尺寸 [batch_size, emb_dim]
                # logits尺寸 [batch_size, category]
                features, logits = model.latent_forward(distilled_samples[j].unsqueeze(0), noise_attention_mask[j].unsqueeze(0), noise_token_type_ids[j].unsqueeze(0))
                # activated_preds = logits.softmax(dim=1)
                activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                _, preds = torch.max(activated_preds, dim=1)
                # batch_loss尺寸 [batch_size]
                batch_loss = criterion(activated_preds, distilled_labels[j].unsqueeze(0))

                loss = torch.sum(batch_loss) / 1
                if (i==Ed-1) and (j==Sd-1):
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                optimizer.step()
                # 清空模型梯度
                model.zero_grad()

        # 清空优化器梯度
        optimizer.zero_grad()
        # 原论文 Algorithm 1 Line 24-25，更新蒸馏样本
        model.eval()
        for batch in client_i_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # labels尺寸 [batch_size]
            labels = batch["labels"].to(device)
            # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
            true_batch_size = labels.size()[0]
            # for param in model.parameters():
            #     param.requires_grad = False
            features, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # activated_preds = logits.softmax(dim=1)
            activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
            _, preds = torch.max(activated_preds, dim=1)
            # batch_loss尺寸 [batch_size]
            batch_loss = criterion(activated_preds, labels)
            loss = batch_loss.mean()
            loss.backward()
            # 只取一个batch，用来构建运算图
            break
        x_grad = distilled_samples.grad
        distilled_samples.data -= alpha * x_grad
        # learning_rate_grad = distilled_learning_rate.grad
        # distilled_learning_rate -= alpha * learning_rate_grad

    return distilled_samples, noise_attention_mask, noise_token_type_ids, distilled_labels


def GroupConstrain_DISTILLDATA(param_dict, model, client_i_dataloader, device,
                               global_group_loss_gap=0.1, probability_difference_threshold=0.2):
    # 自加新参数
    client_batch_num = math.ceil(len(client_i_dataloader.dataset.indices)/param_dict['batch_size'])
    N_of_max_synthetic_batch = 50
    if 70 > client_batch_num > 50:
        N_of_max_synthetic_batch = client_batch_num
    if client_batch_num > 70:
        N_of_max_synthetic_batch = 70

    # N_of_max_synthetic_batch = max(50, math.ceil(len(client_i_dataloader.dataset.indices)/param_dict['batch_size']))
    M_of_min_distilation_step = 2

    # DOSFL超参数
    Ed = 50
    Sd = param_dict['batch_size']
    emb_dim = 768
    distilled_learning_rate = param_dict['learning_rate']
    alpha = 0.1
    τ = 10
    Generator = LatentGenerator(emb_dim).to(device)
    for param in Generator.parameters():
        param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method="DOSFL_ADAM", learning_rate=distilled_learning_rate, decay_steps=τ, max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))
    # 蒸馏出的样本列表
    distilled_samples_list, noise_attention_mask_list, noise_token_type_ids_list, distilled_labels_list = [], [], [], []
    # 记录群组损失差异
    accumulated_group_avg_loss_gap_list = []
    accumulated_group_1_loss = 0
    accumulated_group_0_loss = 0
    accumulated_group_1_count = 0
    accumulated_group_0_count = 0
    # 记录客户预测概率偏差
    client_probability_difference = 0

    # 尝试开始生成蒸馏数据
    for N in range(N_of_max_synthetic_batch):
        logger.info(f"distilling batch: {N}")
        tmp_noise_inputs_embeds = torch.rand([Sd, param_dict['max_len'], emb_dim], device=device)
        tmp_distilled_samples = Generator(tmp_noise_inputs_embeds).to(device)
        tmp_distilled_samples.requires_grad = True
        tmp_distilled_samples.retain_grad()

        tmp_noise_attention_mask = torch.tensor(
            [[1 for i in range(param_dict['max_len'])] for j in range(Sd)], device=device)
        tmp_noise_token_type_ids = torch.tensor(
            [[0 for i in range(param_dict['max_len'])] for j in range(Sd)], device=device)
        tmp_distilled_labels = torch.round(torch.rand(Sd, device=device)).long()

        # 挑选好的样本,易于区分的样本用于被虚拟样本学习，易混淆的样本我们不学
        model.eval()
        target_batch = []
        for iter, batch in enumerate(client_i_dataloader):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # logits尺寸 [batch_size, category]
                features, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                softmax_result = torch.softmax(logits, dim=1)
                label_0_probability_mean = softmax_result[:, 0].mean().item()
                label_1_probability_mean = softmax_result[:, 1].mean().item()
                batch_leve_probability_difference = abs(label_0_probability_mean - label_1_probability_mean)
                torch.cuda.empty_cache()
                # 对于样本质量的评判，可以用概率差可以用信息熵
                if batch_leve_probability_difference >= probability_difference_threshold:
                    target_batch.append((batch, batch_leve_probability_difference))
                    # logger.info(f"batch_leve_probability_difference: {batch_leve_probability_difference}")
                    client_probability_difference += batch_leve_probability_difference
                    break
        # 如果发现样本质量对于现有模型来说不够好,则允许有最低蒸馏次数。如果都失效了就直接不学了
        if len(target_batch) == 0 and N > M_of_min_distilation_step:
            break

        # 如果找到了高质量的样本集合了，就开始蒸馏
        # 先把模型拟合到混乱的虚拟数据中
        model.train()
        logger.info("Update the model with synthetic sample....")
        for i in range(0, Ed):
            # features尺寸 [batch_size, emb_dim]
            # logits尺寸 [batch_size, category]
            features, logits = model.latent_forward(tmp_distilled_samples, tmp_noise_attention_mask, tmp_noise_token_type_ids)
            # activated_preds = logits.softmax(dim=1)
            activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
            _, preds = torch.max(activated_preds, dim=1)
            # batch_loss尺寸 [batch_size]
            batch_loss = criterion(activated_preds, tmp_distilled_labels)
            loss = torch.mean(batch_loss)
            loss.backward()
            optimizer.step()
            # 清空模型梯度
            model.zero_grad()
        optimizer.zero_grad()  # 清空优化器梯度
        # 开始更新蒸馏样本
        model.eval()
        torch.cuda.empty_cache()
        logger.info("Update the synthetic sample with gradient")
        accumulated_group_avg_loss_gap = 0
        # 用更新过后，拟合了虚拟数据分布的模型在目标数据上进行推理，得到损失new_batch_loss
        for b in target_batch:
            with torch.no_grad():
                input_ids = b[0]["input_ids"].to(device)
                attention_mask = b[0]["attention_mask"].to(device)
                labels = b[0]["labels"].to(device)
                true_batch_size = labels.size()[0]
                protecteds = b[0]["protected"].to(device)
                group_flag = protecteds.gt(0.5)
                features, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                new_batch_loss = criterion(logits, labels).detach()
            # 随后再根据new_batch_loss，对虚拟数据进行求梯度操作
            tmp_features, tmp_logits = model.latent_forward(tmp_distilled_samples, tmp_noise_attention_mask, tmp_noise_token_type_ids) # 首先，必须用虚拟数据从新推理一次生成计算图
            tmp_batch_loss = criterion(tmp_logits, tmp_distilled_labels)
            tmp_loss = tmp_batch_loss.sum() / tmp_batch_loss.sum().item() * new_batch_loss.mean().item() # 这里尝试过把原始损失乘以0然后再用加法加上新的损失，发现取出来的梯度为0，所以改了这个乘除法的形式
            tmp_distilled_samples_gradients = autograd.grad(outputs=tmp_loss, inputs=tmp_distilled_samples)[0] # 随后再根据new_batch_loss，对虚拟数据进行求梯度操作
            '''
            DOSFL是参考了Dataset Distillation Using Gradient Matching的思想
            Different from performance matching, which tunes the efficacy of models using synthetic datasets,
            gradient matching refines the performance of networks trained on both the original and synthetic datasets by aligning their training gradients.
            上面这段话的相关引用文献：
            CVPR2024 Workshop https://arxiv.org/pdf/2404.17732
            '''
            # 更新虚拟样本
            tmp_distilled_samples.data -= alpha * tmp_distilled_samples_gradients
            # 清空梯度
            model.zero_grad()
            del tmp_distilled_samples_gradients
            torch.cuda.empty_cache()
            # 引入群组损失差异(对齐)限制--群组梯度差异，以控制到底要蒸馏多少批 样本
            group_1_count = int(sum(group_flag))
            group_0_count = int(true_batch_size - sum(group_flag))
            accumulated_group_1_count += group_1_count
            accumulated_group_0_count += group_0_count
            group_1_sum_loss = float(sum(new_batch_loss[group_flag]))
            group_0_sum_loss = float((sum(new_batch_loss) - sum(new_batch_loss[group_flag])))
            accumulated_group_1_loss += group_1_sum_loss
            accumulated_group_0_loss += group_0_sum_loss
            try:
                accumulated_group_1_avg_loss = float(accumulated_group_1_loss / accumulated_group_1_count)
            except ZeroDivisionError:
                accumulated_group_1_avg_loss = 0.
            try:
                accumulated_group_0_avg_loss = float(accumulated_group_0_loss / accumulated_group_0_count)
            except ZeroDivisionError:
                accumulated_group_0_avg_loss = 0.
            accumulated_group_avg_loss_gap = abs(accumulated_group_0_avg_loss - accumulated_group_1_avg_loss)
            accumulated_group_avg_loss_gap_list.append(accumulated_group_avg_loss_gap)
            distilled_samples_list.append(tmp_distilled_samples.cpu())
            noise_attention_mask_list.append(tmp_noise_attention_mask.cpu())
            noise_token_type_ids_list.append(tmp_noise_token_type_ids.cpu())
            distilled_labels_list.append(tmp_distilled_labels.cpu())

            # 群组损失差异满足范围，则可以停止增加新的虚拟数据批了
            if accumulated_group_avg_loss_gap <= global_group_loss_gap:
                break
            del tmp_distilled_samples, tmp_noise_attention_mask, tmp_noise_token_type_ids, tmp_distilled_labels
            gc.collect()
            torch.cuda.empty_cache()

    if len(distilled_samples_list) != 0:
        group_avg_loss_gap = sum(accumulated_group_avg_loss_gap_list) / len(accumulated_group_avg_loss_gap_list)
        logger.info(f"group_avg_loss_gap: {group_avg_loss_gap}, global_group_loss_gap: {global_group_loss_gap}")

        distilled_samples = torch.concatenate(distilled_samples_list)
        noise_attention_mask = torch.concatenate(noise_attention_mask_list)
        noise_token_type_ids = torch.concatenate(noise_token_type_ids_list)
        distilled_labels = torch.concatenate(distilled_labels_list)

        return distilled_samples, noise_attention_mask, noise_token_type_ids, distilled_labels, group_avg_loss_gap, client_probability_difference/N
    else:
        return None, None, None, None, 0, 0



# Progressive Distilation + Post Train + GroupProto + GroupAlign
def EMAProto_Progressive(device,
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
    del training_dataset, client_dataset_list
    gc.collect()

    # 自定义初始参数
    do_aggregation_flag = False
    # 不做聚合的优势可以引用这篇文章：
    # FedSA: A Unifed Representation Learning via Semantic Anchors for Prototype-based Federated Learning
    do_prototype_flag = True
    do_portrayal_flag = False

    # global_group_loss_gap = param_dict['global_group_loss_gap']
    global_group_loss_gap = 0.1
    # global_probability_gap = param_dict['global_probability_gap']
    global_probability_gap = 0.2

    # 引入新参数
    if do_portrayal_flag:
        # 语义画像列表，长度为用户数目+1，包含所有客户实时更新的画像+最后一个是全局的语义画像
        # 每个项是一个列表，包含的是群组和类排列组合后的prototype
        # [Label 0 Group 0 Proto, Label 0, Group 1 Proto, Label 1, Group 0 Proto, Label 1, Group 1 Proto]
        portrayal_list = [[0, 0, 0, 0] for _ in range(num_clients_K + 1)]
        # 全局与局部语义画像相似度列表，用于后续计算ensemble的权重
        # 如果不计算Prototype，这个列表就为全1，计算出来的ensemble权重就是平均
        global_local_portrayal_similarity_list = [1 for _ in range(num_clients_K)]


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
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")


    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
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

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")
        local_group_loss_gap_list = []
        local_probability_gap_list = []

        if do_prototype_flag:
            global_group_0_label_0_prototype_list = []
            global_group_1_label_0_prototype_list = []
            global_group_0_label_1_prototype_list = []
            global_group_1_label_1_prototype_list = []

            global_group_0_label_0_feature_list = []
            global_group_1_label_0_feature_list = []
            global_group_0_label_1_feature_list = []
            global_group_1_label_1_feature_list = []

            weighted_global_group_0_label_0_feature_list = []
            weighted_global_group_1_label_0_feature_list = []
            weighted_global_group_0_label_1_feature_list = []
            weighted_global_group_1_label_1_feature_list = []

        distilled_samples_list = []
        distilled_labels_list = []

        # Simulate Client Parallel
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            # 下发模型
            logger.info("Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            if do_prototype_flag:
                client_i_group_0_label_0_feature_list = []
                client_i_group_1_label_0_feature_list = []

                client_i_group_0_label_1_feature_list = []
                client_i_group_1_label_1_feature_list = []

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                for batch_id, batch in enumerate(client_i_dataloader):
                # for batch in client_i_dataloader:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # labels尺寸 [batch_size]
                    labels = batch["labels"].to(device)
                    # protected_label尺寸 [batch_size]
                    protecteds = batch["protected"]

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

                    # 记录GPU计算开始时间
                    gpu_start_time = time.time()

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

                    loss = torch.sum(batch_loss) / true_batch_size

                    # 引入了群组损失差异(对齐)限制--群组梯度差异
                    group_flag = protecteds.gt(0.5)
                    one_batch_group_1_count = sum(group_flag)
                    one_batch_group_0_count = true_batch_size - sum(group_flag)
                    if (one_batch_group_1_count != 0) and (one_batch_group_0_count != 0):
                        one_batch_group_1_avg_loss = float(sum(batch_loss[group_flag]) / one_batch_group_1_count)
                        one_batch_group_0_avg_loss = float((sum(batch_loss) - sum(batch_loss[group_flag])) / one_batch_group_0_count)
                        one_batch_group_avg_loss_gap = abs(one_batch_group_0_avg_loss - one_batch_group_1_avg_loss)
                        # logger.info(f"one_batch_group_avg_loss_gap: {one_batch_group_avg_loss_gap} "
                        #             f"in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")
                        loss += one_batch_group_avg_loss_gap

                    # 引入GroupProtoLoss 和 global_GroupProto4CLFLoss 和 local_GroupProto4CLFLoss
                    # 基于分类的两个损失global_GroupProto4CLFLoss 和 local_GroupProto4CLFLoss，启发于2025AAAI
                    # FedSA: A Unifed Representation Learning via Semantic Anchors for Prototype-based Federated Learning
                    # 这篇文章提到3个Observations：
                    # 1 数据分布的偏移(Data Distribution Drift) 会 导致表征不一致(Representation inconsistency for the same input)，即我理解的表征空间的偏移(Drift in Representation Space)
                    # 2 从而进一步导致倾斜的原型对齐(Skewed prototype alignment) 以及 分类器分离(Classifier divergence)，
                    # 原型对齐(Skewed prototype alignment)意味着全局级别不同类别的原型之间的区分度降低
                    # 分类器分离(Classifier divergence)意味着不同客户的分类边界将受到影响
                    #
                    # Prototype的引入以及FedProto的成功已经证实这种方案可以在一定程度上缓解了表征空间的偏移的问题了
                    # 对于减缓分类器的分离，我们借鉴了这篇文章里面的Anchor-based classifer calibration的思路，同时也是考虑了FedProto中没有针对clf进行优化的不足，我们引入了 global_GroupProto4CLFLoss 和 local_GroupProto4CLFLoss
                    #
                    # 同时，为了避免全局的Prototype受到个别bias的客户的过于剧烈的影响，我们采用了EMA的模式更新全局Proto，这可以通过损失等间接且不影响隐私表现的指标来升级成更复杂的更新形式
                    if do_prototype_flag:
                        with torch.no_grad():
                            # 添加原型素材
                            sent_label_flag = labels.gt(0.5).float().reshape([-1, 1]).to(device)
                            sent_group_flag = protecteds.gt(0.5).float().reshape([-1, 1]).to(device)

                            client_i_group_1_label_1_feature_in_one_batch = sent_group_flag * sent_label_flag * features
                            client_i_group_0_label_1_feature_in_one_batch = (1 - sent_group_flag) * sent_label_flag * features
                            client_i_group_1_label_0_feature_in_one_batch = sent_group_flag * (1 - sent_label_flag) * features
                            client_i_group_0_label_0_feature_in_one_batch = (1 - sent_group_flag) * (1 - sent_label_flag) * features

                            client_i_group_1_label_1_feature_list.append(client_i_group_1_label_1_feature_in_one_batch)
                            client_i_group_0_label_1_feature_list.append(client_i_group_0_label_1_feature_in_one_batch)
                            client_i_group_1_label_0_feature_list.append(client_i_group_1_label_0_feature_in_one_batch)
                            client_i_group_0_label_0_feature_list.append(client_i_group_0_label_0_feature_in_one_batch)


                            # 计算Proto gap和 Proto Alignment gap
                            # 每种原型的局部与全局差异
                            (group_0_label_0_feature_gap, group_0_label_1_feature_gap) = 0, 0
                            (group_1_label_0_feature_gap, group_1_label_1_feature_gap) = 0, 0
                            # 每种局部原型的群组差异
                            (label_0_feature_gap, label_1_feature_gap) = 0, 0
                            # 每种全局原型的分类差异
                            (global_group_0_label_0_clf_loss, global_group_0_label_1_clf_loss) = 0, 0
                            (global_group_1_label_0_clf_loss, global_group_1_label_1_clf_loss) = 0, 0

                            (local_group_0_label_0_clf_loss, local_group_0_label_1_clf_loss) = 0, 0
                            (local_group_1_label_0_clf_loss, local_group_1_label_1_clf_loss) = 0, 0
                            # Label 0, Group 0
                            if len(global_group_0_label_0_prototype_list) != 0:
                                client_i_group_0_label_0_one_batch_proto = client_i_group_0_label_0_feature_in_one_batch.mean(dim=0)
                                # 全局-本地类原型的差异（欧几里得距离）
                                g = global_group_0_label_0_prototype_list[-1].cuda()
                                l = client_i_group_0_label_0_one_batch_proto.cuda()
                                label_0_feature_gap = torch.norm(g-l, p=2)
                                group_0_label_0_feature_gap = 0.5 * float(label_0_feature_gap ** 2)

                                tmp_label = torch.tensor([1,0]).float()
                                # 全局原型的训练损失
                                __, tmp_logit = model.only_clf_forward(g)
                                global_group_0_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()
                                # 局部原型的训练损失
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_0_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()

                                logger.info(f"group_0_label_0_feature_gap：{group_0_label_0_feature_gap} and "
                                            f"global_group_0_label_0_clf_loss: {global_group_0_label_0_clf_loss} and "
                                            f"local_group_0_label_0_clf_loss: {local_group_0_label_0_clf_loss} "
                                            f"in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")

                                del client_i_group_0_label_0_one_batch_proto

                            # Label 0, Group 1
                            if len(global_group_1_label_0_prototype_list) != 0:
                                client_i_group_1_label_0_one_batch_proto = client_i_group_1_label_0_feature_in_one_batch.mean(dim=0)
                                # 全局-本地类原型的差异（欧几里得距离）
                                g = global_group_1_label_0_prototype_list[-1].cuda()
                                l = client_i_group_1_label_0_one_batch_proto.cuda()
                                label_0_feature_gap = torch.norm(g-l, p=2)
                                group_1_label_0_feature_gap = 0.5 * float(label_0_feature_gap ** 2)

                                tmp_label = torch.tensor([1,0]).float()
                                # 全局原型的训练损失
                                __, tmp_logit = model.only_clf_forward(g)
                                global_group_1_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()
                                # 局部原型的训练损失
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_1_label_0_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()

                                logger.info(f"group_1_label_0_feature_gap: {group_1_label_0_feature_gap} and "
                                            f"global_group_1_label_0_clf_loss: {global_group_1_label_0_clf_loss} and "
                                            f"local_group_1_label_0_clf_loss: {local_group_1_label_0_clf_loss} "
                                            f"in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")

                                del client_i_group_1_label_0_one_batch_proto

                            # Label 1, Group 0
                            if len(global_group_0_label_1_prototype_list) != 0:
                                client_i_group_0_label_1_one_batch_proto = client_i_group_0_label_1_feature_in_one_batch.mean(dim=0)
                                # 全局-本地类原型的差异（欧几里得距离）
                                g = global_group_0_label_1_prototype_list[-1].cuda()
                                l = client_i_group_0_label_1_one_batch_proto.cuda()
                                label_1_feature_gap = torch.norm(g-l, p=2)
                                group_0_label_1_feature_gap = 0.5 * float(label_1_feature_gap ** 2)

                                __, tmp_logit = model.only_clf_forward(g)
                                # 全局原型的训练损失
                                tmp_label = torch.tensor([0,1]).float()
                                global_group_0_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()
                                # 局部原型的训练损失
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_0_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()

                                logger.info(f"group_0_label_1_feature_gap: {group_0_label_1_feature_gap} and "
                                            f"global_group_0_label_1_clf_loss: {global_group_0_label_1_clf_loss} and "
                                            f"local_group_0_label_1_clf_loss: {local_group_0_label_1_clf_loss} "
                                            f"in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")
                                del client_i_group_0_label_1_one_batch_proto

                            # Label 1, Group 1
                            if len(global_group_1_label_1_prototype_list) != 0:
                                client_i_group_1_label_1_one_batch_proto = client_i_group_1_label_1_feature_in_one_batch.mean(dim=0)
                                # 全局-本地类原型的差异（欧几里得距离）
                                g = global_group_1_label_1_prototype_list[-1].cuda()
                                l = client_i_group_1_label_1_one_batch_proto.cuda()
                                label_1_feature_gap = torch.norm(g-l, p=2)
                                group_1_label_1_feature_gap = 0.5 * float(label_1_feature_gap ** 2)

                                tmp_label = torch.tensor([0, 1]).float()
                                # 全局原型的训练损失
                                __, tmp_logit = model.only_clf_forward(g)
                                global_group_1_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()
                                # 局部原型的训练损失
                                __, tmp_logit = model.only_clf_forward(l)
                                local_group_1_label_1_clf_loss = criterion(tmp_logit.to(device), tmp_label.to(device)).mean()

                                logger.info(f"group_1_label_1_feature_gap: {group_1_label_1_feature_gap} and "
                                            f"global_group_1_label_1_clf_loss: {global_group_1_label_1_clf_loss} and "
                                            f"local_group_1_label_1_clf_loss: {local_group_1_label_1_clf_loss} "
                                            f"in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")

                                del client_i_group_1_label_1_one_batch_proto

                            lamda_list = [1, 1, 1, 1,
                                          1, 1,
                                          1, 1, 1, 1,
                                          1, 1, 1, 1,]  # FedPro思路
                            reg_list = [group_0_label_0_feature_gap, group_0_label_1_feature_gap,
                                        group_1_label_0_feature_gap, group_1_label_1_feature_gap,
                                        0, 0,
                                        global_group_0_label_0_clf_loss, global_group_0_label_1_clf_loss,
                                        global_group_1_label_0_clf_loss, global_group_1_label_1_clf_loss,
                                        local_group_0_label_0_clf_loss, local_group_0_label_1_clf_loss,
                                        local_group_1_label_0_clf_loss, local_group_1_label_1_clf_loss,
                                        ]
                            for index, lamda in enumerate(lamda_list):
                                loss += lamda * reg_list[index]


                            del sent_label_flag, sent_group_flag
                            del client_i_group_1_label_1_feature_in_one_batch, client_i_group_0_label_1_feature_in_one_batch
                            del client_i_group_1_label_0_feature_in_one_batch, client_i_group_0_label_0_feature_in_one_batch


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

                    del input_ids, attention_mask, labels, batch_loss, loss
                    gc.collect()
                    torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

                # logger.debug(f"GPU Memory :")
                # logger.debug(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                gc.collect()

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            # 本地更新完全后，做一次数据集蒸馏
            # 记录GPU计算开始时间
            gpu_start_time = time.time()

            if do_prototype_flag:
                with torch.no_grad():
                    # 计算客户的 类原型
                    group_0_label_0_flag = len(client_i_group_0_label_0_feature_list) != 0
                    group_1_label_0_flag = len(client_i_group_1_label_0_feature_list) != 0
                    group_0_label_1_flag = len(client_i_group_0_label_1_feature_list) != 0
                    group_1_label_1_flag = len(client_i_group_1_label_1_feature_list) != 0

                    # Label 0, Group 0
                    if group_0_label_0_flag:
                        # 得到客户的原型
                        # client_i_group_0_label_0_prototype = torch.stack(client_i_group_0_label_0_feature_list, dim=0).mean(dim=0)
                        client_i_group_0_label_0_prototype = torch.concatenate(client_i_group_0_label_0_feature_list, dim=0).mean(dim=0)
                        client_i_label_0_proto = client_i_group_0_label_0_prototype
                        if do_portrayal_flag:
                            # 更新用户画像
                            logger.info(f"Update Client {id} 's portrayal (Label 0, Group 0)")
                            portrayal_list[id][0] = client_i_group_0_label_0_prototype

                        global_group_0_label_0_feature_list.append(client_i_group_0_label_0_prototype)
                        # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                        weighted_global_group_0_label_0_feature_list.append(client_i_aggregation_weight * client_i_group_0_label_0_prototype)
                    # Label 0, Group 1
                    if group_1_label_0_flag:
                        # 得到客户的原型
                       # client_i_group_1_label_0_prototype = torch.stack(client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)
                        client_i_group_1_label_0_prototype = torch.concatenate(client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)
                        client_i_label_0_proto = client_i_group_1_label_0_prototype

                        if do_portrayal_flag:
                            # 更新用户画像
                            logger.info(f"Update Client {id} 's portrayal (Label 0, Group 1)")
                            portrayal_list[id][1] = client_i_group_1_label_0_prototype

                        global_group_1_label_0_feature_list.append(client_i_group_1_label_0_prototype)
                        # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                        weighted_global_group_1_label_0_feature_list.append(client_i_aggregation_weight * client_i_group_1_label_0_prototype)
                    # Label 1, Group 0
                    if group_0_label_1_flag:
                        # 得到客户的原型
                        # client_i_group_0_label_1_prototype = torch.stack(client_i_group_0_label_1_feature_list, dim=0).mean(dim=0)
                        client_i_group_0_label_1_prototype = torch.concatenate(client_i_group_0_label_1_feature_list, dim=0).mean(dim=0)
                        client_i_label_1_proto = client_i_group_0_label_1_prototype

                        if do_portrayal_flag:
                            # 更新用户画像
                            logger.info(f"Update Client {id} 's portrayal (Label 1, Group 0)")
                            portrayal_list[id][2] = client_i_group_0_label_1_prototype

                        global_group_0_label_1_feature_list.append(client_i_group_0_label_1_prototype)
                        # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                        weighted_global_group_0_label_1_feature_list.append(client_i_aggregation_weight * client_i_group_0_label_1_prototype)
                    # Label 1, Group 1
                    if group_1_label_1_flag:
                        # 得到客户的原型
                        # client_i_group_1_label_1_prototype = torch.stack(client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
                        client_i_group_1_label_1_prototype = torch.concatenate(client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
                        client_i_label_1_proto = client_i_group_1_label_1_prototype

                        if do_portrayal_flag:
                            # 更新用户画像
                            logger.info(f"Update Client {id} 's portrayal (Label 1, Group 1)")
                            portrayal_list[id][3] = client_i_group_1_label_1_prototype

                        global_group_1_label_1_feature_list.append(client_i_group_1_label_1_prototype)
                        # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                        weighted_global_group_1_label_1_feature_list.append(client_i_aggregation_weight * client_i_group_1_label_1_prototype)

                    # Update Label 0
                    if group_0_label_0_flag and group_1_label_0_flag:
                        # client_i_label_0_proto = torch.stack(client_i_group_0_label_0_feature_list + client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)
                        client_i_label_0_proto = torch.concatenate(client_i_group_0_label_0_feature_list + client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)

                    # Update Label 1
                    if group_0_label_1_flag and group_1_label_1_flag:
                        # client_i_label_0_proto = torch.stack(client_i_group_0_label_1_feature_list + client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
                        client_i_label_0_proto = torch.concatenate(client_i_group_0_label_1_feature_list + client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)


            # 本地数据蒸馏
            logger.info(f"Client {id} distilling local data")
            # distilled_inputs_embeds, distilled_noise_attention_mask, distilled_noise_token_type_ids, distilled_labels = DOSFL_DISTILLDATA(param_dict, model, client_i_dataloader,device)
            (distilled_inputs_embeds, distilled_noise_attention_mask, distilled_noise_token_type_ids, distilled_labels,
             local_group_loss_gap, local_probability_gap) = GroupConstrain_DISTILLDATA(param_dict, model, client_i_dataloader, device, global_group_loss_gap, global_probability_gap)
            local_group_loss_gap_list.append(client_i_aggregation_weight *local_group_loss_gap)
            local_probability_gap_list.append(client_i_aggregation_weight *local_probability_gap)
            if distilled_inputs_embeds != None:
                distilled_samples_list.append([distilled_inputs_embeds.cpu(), distilled_noise_attention_mask.cpu(), distilled_noise_token_type_ids.cpu()])
                distilled_labels_list.append(distilled_labels.cpu())

            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            del distilled_inputs_embeds, distilled_noise_attention_mask, distilled_noise_token_type_ids, distilled_labels
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)


        # Global operation
        # 更新delta
        global_group_loss_gap = sum(local_group_loss_gap_list)  # 前面已经乘过权重（client_i_aggregation_weight）了，所以这里只需要加起来即可得到全局的delta
        global_probability_gap = sum(local_probability_gap_list)  # 前面已经乘过权重（client_i_aggregation_weight）了，所以这里只需要加起来即可得到全局的delta

        logger.info(f"Communication Round {(iter_t + 1)} "
                    f"Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB, "
                    f"global_group_loss_gap: {global_group_loss_gap}, "
                    f"global_probability_gap: {global_probability_gap} ")

        if do_prototype_flag:
            # 更新全局原型和语义画像
            logger.info("Prototype aggregation and portrayal update")
            (global_group_0_label_0_prototype, global_group_0_label_1_prototype) = 0, 0
            (global_group_1_label_0_prototype, global_group_1_label_1_prototype) = 0, 0

            # 前面已经乘过权重（client_i_aggregation_weight）了，所以这里只需要加起来即可得到全局的prototype
            # Label 0, Group 0
            if len(weighted_global_group_0_label_0_feature_list) != 0:
                for proto in weighted_global_group_0_label_0_feature_list:
                    global_group_0_label_0_prototype += proto
                if do_portrayal_flag:
                    logger.info("Update global portrayal (Label 0, Group 0)")
                    portrayal_list[-1][0] = global_group_0_label_0_prototype
                # 引入EMA式的全局Prototype更新
                if len(global_group_0_label_0_prototype_list) != 0:
                    global_group_0_label_0_prototype_list.append(
                        0.1 * global_group_0_label_0_prototype_list[-1] + 0.9 * global_group_0_label_0_prototype
                    )
                else:
                    global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
            # Label 0, Group 1
            if len(weighted_global_group_1_label_0_feature_list) != 0:
                for proto in weighted_global_group_1_label_0_feature_list:
                    global_group_1_label_0_prototype += proto
                if do_portrayal_flag:
                    logger.info("Update global portrayal (Label 0, Group 1)")
                    portrayal_list[-1][1] = global_group_1_label_0_prototype
                # 引入EMA式的全局Prototype更新
                if len(global_group_1_label_0_prototype_list) != 0:
                    global_group_0_label_0_prototype_list.append(
                        0.1 * global_group_1_label_0_prototype_list[-1] + 0.9 * global_group_1_label_0_prototype
                    )
                else:
                    global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
            # Label 1, Group 0
            if len(weighted_global_group_0_label_1_feature_list) != 0:
                for proto in weighted_global_group_0_label_1_feature_list:
                    global_group_0_label_1_prototype += proto
                if do_portrayal_flag:
                    logger.info("Update global portrayal (Label 1, Group 0)")
                    portrayal_list[-1][2] = global_group_0_label_1_prototype
                global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
            # Label 1, Group 1
            if len(weighted_global_group_1_label_1_feature_list) != 0:
                for proto in weighted_global_group_1_label_1_feature_list:
                    global_group_1_label_1_prototype += proto
                if do_portrayal_flag:
                    logger.info("Update global portrayal (Label 1, Group 1)")
                    portrayal_list[-1][3] = global_group_1_label_1_prototype
                global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型

            if do_portrayal_flag:
                # 更新语义画像相似度
                logger.info("Portrayal Semantic Similarity update")
                try:
                    global_portrayal = portrayal_list[-1]
                    for id in idxs_users:
                        client_portrayal = portrayal_list[id]
                        sim_0 = torch.cosine_similarity(global_portrayal[0], client_portrayal[0])
                        sim_1 = torch.cosine_similarity(global_portrayal[1], client_portrayal[1])
                        sim_2 = torch.cosine_similarity(global_portrayal[2], client_portrayal[2])
                        sim_3 = torch.cosine_similarity(global_portrayal[3], client_portrayal[3])
                        local_global_sim = (sim_0 + sim_1 + sim_2 + sim_3) / 4
                        global_local_portrayal_similarity_list[id] = local_global_sim
                except Exception:
                    continue

        if do_aggregation_flag:
            logger.info("Parameter aggregation")
            theta_list = []
            for id in idxs_users:
                client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
                selected_model = torch.load(client_model_path)  # 持久化
                theta_list.append(get_parameters(selected_model))
                del selected_model
                gc.collect()

            theta_list = np.array(theta_list, dtype=object)
            # FedAvg新版论文的聚合权重是数据占比
            # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
            # 如一个weights = [w1, w2, w3, w4]
            # 那么结果就是(theta1 * w1 + theta2 * w2 + theta3 * w3 + theta4 * w4)/ sum(w1+w2+w3+w4)
            theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
            # FedAvg旧版论文的聚合权重是平均
            # theta_avg = np.mean(theta_list, 0).tolist()

            logger.info("Update Global Model with aggregated parameters")
            set_parameters(global_model, theta_avg)

            del theta_list
            gc.collect()

        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        # 聚合权重后继续做对齐训练
        global_model = global_model.to(device)
        global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                             learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        global_optimizer.set_parameters(list(global_model.named_parameters()))


        # 先做输出分布对齐训练，以等效替代参数聚合的环节，让全局模型对齐局部模型
        global_criterion = torch.nn.MSELoss()

        accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
        logger.info(f"Performance before post training: ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")

        logger.info(f"Sever side post training phase 1: PoTrain with Synthetic Samples")

        max_phase1_post_train_step = max(50, math.ceil(np.array(client_datasets_size_list)[idxs_users].mean()/param_dict['batch_size']))
        for _ in range(max_phase1_post_train_step):
            with torch.no_grad():
                emb_dim = 768
                logger.info(f"Create Synthetic Samples")
                # batch_noise尺寸[batch_size, seq_length, embedding_dim]
                batch_noise_inputs_embeds = torch.rand([param_dict['batch_size'], param_dict['max_len'], emb_dim])
                batch_noise_attention_mask = torch.tensor(
                    [[1 for i in range(param_dict['max_len'])] for j in range(param_dict['batch_size'])])
                batch_noise_token_type_ids = torch.tensor(
                    [[0 for i in range(param_dict['max_len'])] for j in range(param_dict['batch_size'])])

                batch_noise_inputs_embeds = batch_noise_inputs_embeds.to(device)
                batch_noise_attention_mask = batch_noise_attention_mask.to(device)
                batch_noise_token_type_ids = batch_noise_token_type_ids.to(device)

                logger.info(f"Using Client Models to Inference")
                client_feature_list = []
                client_logit_list = []
                if do_portrayal_flag:
                    selected_client_similarity = [global_local_portrayal_similarity_list[id] for id in idxs_users]
                else:
                    selected_client_similarity = [1 for _ in idxs_users]
                tmp_sum = sum(selected_client_similarity)
                for index, id in enumerate(idxs_users):
                    ensemble_weighted = selected_client_similarity[index] / tmp_sum
                    client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
                    client_model = torch.load(client_model_path)
                    client_model.eval()
                    client_model = client_model.to(device)
                    # client_feature [batch_size, emb_dim]
                    # client_logit尺寸[batch_size, category]
                    client_feature, client_logit = client_model.latent_forward(batch_noise_inputs_embeds, batch_noise_attention_mask,
                                                                  batch_noise_token_type_ids)
                    client_feature_list.append(ensemble_weighted * client_feature.to(device))
                    client_logit_list.append(ensemble_weighted * client_logit.to(device))

                    del client_model, client_feature, client_logit
                    gc.collect()
                    torch.cuda.empty_cache()

                # client_batch_feature尺寸[num_clients_K, batch_size, emb_dim]
                client_batch_feature = torch.stack(client_feature_list).cuda()
                # ensembled_batch_feature尺寸[batch_size, emb_dim]
                ensembled_batch_feature = client_batch_feature.mean(dim=0).to(device)  # 原论文提供的代码就是平均各个client的结果处理

                # client_batch_logit尺寸[num_clients_K, batch_size, category]
                client_batch_logit = torch.stack(client_logit_list).cuda()
                # ensembled_batch_logit尺寸[batch_size, category]
                ensembled_batch_logit = client_batch_logit.mean(dim=0).to(device)  # 原论文提供的代码就是平均各个client的结果处理

                del client_feature_list, client_logit_list
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(f"Using Global Model to Inference")
            global_feature, global_logit = global_model.latent_forward(batch_noise_inputs_embeds, batch_noise_attention_mask,
                                                          batch_noise_token_type_ids)
            global_contrastive_loss = 0

            # 第一种损失：特征级别的对比损失，参考 偏标签的消歧
            # PiCO: Contrastive Label Disambiguation for Partial Label Learning
            # https://zhuanlan.zhihu.com/p/463255610
            # https://hbzju.github.io/pico/
            # _, soft_ensemble_preds = torch.max(ensembled_batch_logit, dim=1)
            # soft_ensemble_flag = soft_ensemble_preds.gt(0.5).int().reshape([-1, 1]).to(device)
            # soft_label_1_global_feature = soft_ensemble_flag * global_feature
            # soft_label_0_global_feature = (1-soft_ensemble_flag) * global_feature
            # soft_lable_0_index = [index for index,item in enumerate(soft_ensemble_flag.reshape([-1])) if item == 0]
            # soft_lable_1_index = [index for index,item in enumerate(soft_ensemble_flag.reshape([-1])) if item == 1]
            #
            # if soft_ensemble_flag.sum().item() == soft_ensemble_preds.shape[0] or soft_ensemble_flag.sum().item() == 0:
            #     global_contrastive_loss = 0
            # else:
            #     try:
            #         label_0_InfoNCE_loss_avg, label_0_InfoNCE_loss_sum = InfoNCE_loss_of_samples(z=soft_label_0_global_feature,
            #                                                                                      pos_indexs=soft_lable_0_index,
            #                                                                                      neg_indexs=soft_lable_1_index)
            #     except Exception:
            #         label_0_InfoNCE_loss_avg, label_0_InfoNCE_loss_sum = 0, 0
            #     try:
            #         label_1_InfoNCE_loss_avg, label_1_InfoNCE_loss_sum = InfoNCE_loss_of_samples(z=soft_label_1_global_feature,
            #                                                                                      pos_indexs=soft_lable_1_index,
            #                                                                                      neg_indexs=soft_lable_0_index)
            #     except Exception:
            #         label_1_InfoNCE_loss_avg, label_1_InfoNCE_loss_sum = 0, 0
            #
            #     global_contrastive_loss = torch.nan_to_num(label_0_InfoNCE_loss_avg, nan=0) + torch.nan_to_num(label_1_InfoNCE_loss_avg, nan=0)

            global_loss_1 = global_criterion(ensembled_batch_logit, global_logit)
            global_loss_2 = global_criterion(ensembled_batch_feature, global_feature)
            logger.info(f" Phase 1 contrastive_loss: {global_contrastive_loss}, logit_align_loss: {round(float(global_loss_1), 3)}, feature_align_loss: {round(float(global_loss_2), 3)}")

            global_loss = global_contrastive_loss + global_loss_1 + global_loss_2

            global_loss.backward()
            global_optimizer.step()
            global_model.zero_grad()
            global_optimizer.zero_grad()

        accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
        logger.info(f" Phase 1 ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
        logger.info(f" ####################################################################################################")

        if len(distilled_samples_list) != 0:
            max_phase2_post_train_step = max(50, math.ceil(np.array(client_datasets_size_list)[idxs_users].mean()/param_dict['batch_size']))
            break_the_post_train_flag = False
            for _ in range(max_phase2_post_train_step):
                if break_the_post_train_flag:
                    break

                logger.info(f"Sever side post training phase 2: PoTrain with distilled Samples")
                # 再做基于蒸馏数据的公平对齐训练
                global_criterion = torch.nn.CrossEntropyLoss(reduction='none')
                random.shuffle(idxs_users)
                tmp_total_loss, tmp_total_count = 0, 0

                for index, distilled_samples in enumerate(distilled_samples_list):

                    distilled_inputs_embeds, distilled_noise_attention_mask, distilled_noise_token_type_ids = distilled_samples
                    distilled_labels = distilled_labels_list[index]
                    distilled_size = len(distilled_labels)
                    for i in range(0, distilled_size, param_dict['batch_size']):
                    # for i in range(0, distilled_size, 256):
                        _, global_logit = global_model.latent_forward(distilled_inputs_embeds[i:i+param_dict['batch_size']].cuda(),
                                                                      distilled_noise_attention_mask[i:i+param_dict['batch_size']].cuda(),
                                                                      distilled_noise_token_type_ids[i:i+param_dict['batch_size']].cuda())
                        if do_portrayal_flag:
                            # 损失由语义画像相似度来缩放控制
                            tmp_loss = global_local_portrayal_similarity_list[id] * global_criterion(global_logit, distilled_labels[i:i+param_dict['batch_size']])
                        else:
                            tmp_loss = global_criterion(global_logit, distilled_labels[i: i+param_dict['batch_size']].cuda())

                        # _, global_logit = global_model.latent_forward(
                        #     distilled_inputs_embeds[i:i + 256].cuda(),
                        #     distilled_noise_attention_mask[i:i + 256].cuda(),
                        #     distilled_noise_token_type_ids[i:i + 256].cuda())
                        # if do_portrayal_flag:
                        #     # 损失由语义画像相似度来缩放控制
                        #     tmp_loss = global_local_portrayal_similarity_list[id] * global_criterion(global_logit,
                        #                                                                              distilled_labels[i:i + 256])
                        # else:
                        #     tmp_loss = global_criterion(global_logit, distilled_labels[i: i + 256].cuda())
                        tmp_count = len(tmp_loss)
                        tmp_total_loss += float(sum(tmp_loss))
                        tmp_total_count += tmp_count
                        global_loss = tmp_loss.sum() / tmp_count
                        global_loss.backward()
                        global_optimizer.step()

                        global_model.zero_grad()
                        global_optimizer.zero_grad()

                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                                   testing_dataset_len)
                logger.info(f" Phase 2 ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}, Avg Loss over samples: {round((tmp_total_loss/tmp_total_count) ,3)}")
                logger.info(f" ####################################################################################################")

        logger.info("Update Global Model with Progressive training")
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
        # if (iter_t + 1) != param_dict['communication_round_I']:
        #     accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
        #     logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_Progressive.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
