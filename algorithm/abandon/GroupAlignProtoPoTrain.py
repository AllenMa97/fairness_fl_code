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
from tool.utils import FL_fairness_and_accuracy_test


def GroupAlignProtoPoTrain(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    gc.collect()

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
            style="FedAvg",
        )
        selected_client_training_dataset_size = sum([client_datasets_size_list[item] for item in idxs_users])
        average_weight = [0 for _ in range(num_clients_K)]
        for id in idxs_users:
            average_weight[id] = client_datasets_size_list[id] / selected_client_training_dataset_size
        average_weight = np.array(average_weight)

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")


        global_group_0_label_0_prototype_list = []
        global_group_1_label_0_prototype_list = []
        global_group_0_label_1_prototype_list = []
        global_group_1_label_1_prototype_list = []

        global_group_0_label_0_feature_list = []
        global_group_1_label_0_feature_list = []
        global_group_0_label_1_feature_list = []
        global_group_1_label_1_feature_list = []


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
                # for batch_index, batch in enumerate(client_i_dataloader):
                for batch in client_i_dataloader:
                    group_0_label_0_feature_list = []
                    group_1_label_0_feature_list = []
                    group_0_label_1_feature_list = []
                    group_1_label_1_feature_list = []


                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # labels尺寸 [batch_size]
                    labels = batch["labels"].to(device)
                    # protected_label尺寸 [batch_size]
                    protecteds = batch["protected"].to(device)

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

                    # 引入GroupProtoLoss
                    # 添加原型素材
                    features = features.cpu()
                    sent_label_flag = labels.gt(0.5).tolist()
                    sent_group_flag = protecteds.gt(0.5).tolist()
                    for idx, feat in enumerate(features):
                        if sent_label_flag[idx]:
                            if sent_group_flag[idx]:
                                group_1_label_1_feature_list.append(feat)
                                client_i_group_1_label_1_feature_list.append(feat)
                            else:
                                group_0_label_1_feature_list.append(feat)
                                client_i_group_0_label_1_feature_list.append(feat)
                        else:
                            if sent_group_flag[idx]:
                                group_1_label_0_feature_list.append(feat)
                                client_i_group_1_label_0_feature_list.append(feat)
                            else:
                                group_0_label_0_feature_list.append(feat)
                                client_i_group_0_label_0_feature_list.append(feat)

                    # 计算Proto gap和 Proto Alignment gap
                    (group_0_label_0_feature_gap, group_0_label_1_feature_gap) = 0, 0
                    (group_1_label_0_feature_gap, group_1_label_1_feature_gap) = 0, 0
                    (label_0_feature_gap, label_1_feature_gap) = 0, 0
                    with torch.no_grad():
                        # 全局-本地类原型的差异
                        # Label 0, Protected 0
                        if len(client_i_group_0_label_0_feature_list) != 0:
                            group_0_label_0_prototype = torch.stack(client_i_group_0_label_0_feature_list, dim=0).mean(dim=0)
                            if len(global_group_0_label_0_prototype_list) != 0:
                                label_0_feature_gap = torch.norm(
                                    (global_group_0_label_0_prototype_list[-1].cuda() - group_0_label_0_prototype.cuda()),
                                    p=2)
                                group_0_label_0_feature_gap = float(label_0_feature_gap)

                        # 全局-本地类原型的差异
                        # Label 0, Protected 1
                        if len(client_i_group_1_label_0_feature_list) != 0:
                            group_1_label_0_prototype = torch.stack(client_i_group_1_label_0_feature_list,
                                                                    dim=0).mean(dim=0)
                            if len(global_group_1_label_0_prototype_list) != 0:
                                label_0_feature_gap = torch.norm(
                                    (global_group_1_label_0_prototype_list[
                                         -1].cuda() - group_1_label_0_prototype.cuda()),
                                    p=2)
                                group_1_label_0_feature_gap = float(label_0_feature_gap)

                        # 同一个label不同group的差异
                        # Label 0, Protected 0, Label 0, Protected 1
                        if len(client_i_group_0_label_0_feature_list) != 0:
                            if len(client_i_group_1_label_0_feature_list) != 0:
                                label_0_feature_gap = torch.norm(
                                    (group_0_label_0_prototype.cuda() - group_1_label_0_prototype.cuda()),
                                    p=2)
                                label_0_feature_gap = float(label_0_feature_gap)

                        # 全局-本地类原型的差异
                        # Label 1, Protected 0
                        if len(client_i_group_0_label_1_feature_list) != 0:
                            group_0_label_1_prototype = torch.stack(client_i_group_0_label_1_feature_list, dim=0).mean(dim=0)
                            if len(global_group_0_label_1_prototype_list) != 0:
                                label_1_feature_gap = torch.norm(
                                    (global_group_0_label_1_prototype_list[-1].cuda() - group_0_label_1_prototype.cuda()),
                                    p=2)
                                group_0_label_1_feature_gap = float(label_1_feature_gap)


                        # 全局-本地类原型的差异
                        # Label 1, Protected 1
                        if len(client_i_group_1_label_1_feature_list) != 0:
                            group_1_label_1_prototype = torch.stack(client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
                            if len(global_group_1_label_1_prototype_list) != 0:
                                label_1_feature_gap = torch.norm(
                                    (global_group_1_label_1_prototype_list[-1].cuda() - group_1_label_1_prototype.cuda()),
                                    p=2)
                                group_1_label_1_feature_gap = float(label_1_feature_gap)

                        # 同一个label不同group的差异
                        # Label 1, Protected 0, Label 1, Protected 1
                        if len(client_i_group_0_label_1_feature_list) != 0:
                            if len(client_i_group_1_label_1_feature_list) != 0:
                                label_1_feature_gap = torch.norm(
                                    (group_0_label_1_prototype.cuda() - group_1_label_1_prototype.cuda()),
                                    p=2)
                                label_1_feature_gap = float(label_1_feature_gap)


                    lamda_list = [1, 1, 1, 1, 1, 1]  # FedPro思路
                    gap_list = [group_0_label_0_feature_gap, group_0_label_1_feature_gap, group_1_label_0_feature_gap, group_1_label_1_feature_gap,
                                label_0_feature_gap, label_1_feature_gap]
                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * gap_list[index]


                    loss.backward()

                    # FedAvg算法一个batch就做一次更新
                    optimizer.step()


                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()

                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 清空梯度
                    model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    del input_ids, attention_mask, labels
                    gc.collect()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化


            # 记录GPU计算开始时间
            gpu_start_time = time.time()
            # 计算客户的 类原型
            # Label 0, Protected 0
            if len(client_i_group_0_label_0_feature_list) != 0:
                client_i_label_0_prototype = torch.stack(client_i_group_0_label_0_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_group_0_label_0_feature_list.append(client_i_aggregation_weight * client_i_label_0_prototype)
            # Label 0, Protected 1
            if len(client_i_group_1_label_0_feature_list) != 0:
                client_i_label_0_prototype = torch.stack(client_i_group_1_label_0_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_group_1_label_0_feature_list.append(client_i_aggregation_weight * client_i_label_0_prototype)
            # Label 1, Protected 0
            if len(client_i_group_0_label_1_feature_list) != 0:
                client_i_label_1_prototype = torch.stack(client_i_group_0_label_1_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_group_0_label_1_feature_list.append(client_i_aggregation_weight * client_i_label_1_prototype)
            # Label 1, Protected 1
            if len(client_i_group_1_label_1_feature_list) != 0:
                client_i_label_1_prototype = torch.stack(client_i_group_1_label_1_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_group_1_label_1_feature_list.append(client_i_aggregation_weight * client_i_label_1_prototype)

            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            del model
            gc.collect()
            # torch.cuda.empty_cache()

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # Global operation

        logger.info("Prototype aggregation")
        (global_group_0_label_0_prototype, global_group_0_label_1_prototype) = 0, 0
        (global_group_1_label_0_prototype, global_group_1_label_1_prototype) = 0, 0
        # 前面已经乘过权重了，所以这里只需要加起来即可

        # Label 0, Protected 0
        if len(global_group_0_label_0_feature_list) != 0:
            for proto in global_group_0_label_0_feature_list:
                global_group_0_label_0_prototype += proto
            global_group_0_label_0_prototype_list.append(global_group_0_label_0_prototype)  # 更新全局的各种原型
        # Label 0, Protected 1
        if len(global_group_1_label_0_feature_list) != 0:
            for proto in global_group_1_label_0_feature_list:
                global_group_1_label_0_prototype += proto
            global_group_1_label_0_prototype_list.append(global_group_1_label_0_prototype)  # 更新全局的各种原型
        # Label 1, Protected 0
        if len(global_group_0_label_1_feature_list) != 0:
            for proto in global_group_0_label_1_feature_list:
                global_group_0_label_1_prototype += proto
            global_group_0_label_1_prototype_list.append(global_group_0_label_1_prototype)  # 更新全局的各种原型
        # Label 1, Protected 1
        if len(global_group_1_label_1_feature_list) != 0:
            for proto in global_group_1_label_1_feature_list:
                global_group_1_label_1_prototype += proto
            global_group_1_label_1_prototype_list.append(global_group_1_label_1_prototype)  # 更新全局的各种原型



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
        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)

        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        # 聚合权重后继续做对齐训练
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
            client_logit_list = []
            for id in idxs_users:
                client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
                client_model = torch.load(client_model_path)
                client_model.eval()
                client_model = client_model.to(device)
                # client_logit尺寸[batch_size, category]
                _, client_logit = client_model.latent_forward(batch_noise_inputs_embeds, batch_noise_attention_mask,
                                                              batch_noise_token_type_ids)
                client_logit_list.append(client_logit.to(device))
                del client_model
                gc.collect()
                torch.cuda.empty_cache()
            # client_batch_logit尺寸[num_clients_K, batch_size, category]
            client_batch_logit = torch.stack(client_logit_list).cuda()
            # ensembled_batch_logit尺寸[batch_size, category]
            ensembled_batch_logit = client_batch_logit.mean(dim=0).to(device)  # 原论文提供的代码就是平均各个client的结果处理

        logger.info(f"Using Global Model to Inference")
        global_model = global_model.to(device)
        _, global_logit = global_model.latent_forward(batch_noise_inputs_embeds, batch_noise_attention_mask,
                                                      batch_noise_token_type_ids)
        global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                             learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        global_optimizer.set_parameters(list(global_model.named_parameters()))

        global_criterion = torch.nn.MSELoss()
        global_loss = global_criterion(ensembled_batch_logit, global_logit)
        global_loss.backward()
        global_optimizer.step()
        logger.info("Update Global Model 2")
        # 记录GPU计算结束时间
        gpu_end_time = time.time()
        total_gpu_seconds += (gpu_end_time - gpu_start_time)


        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I']:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
            logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_GroupProto.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
