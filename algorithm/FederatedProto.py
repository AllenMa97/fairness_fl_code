import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value


def Fed_PROTO(device,
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
    users_gpu_seconds_list = [0] * num_clients_K
    num_of_class = 2
    # prototype_MB_size = sys.getsizeof(torch.rand([num_of_class, 768])) / (1024 ** 2)
    prototype_MB_size = torch.rand([num_of_class, 768]).numel() * 4 / (1024 ** 2)

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

        global_label_0_prototype_list = []
        global_label_1_prototype_list = []

        global_label_0_feature_list = []
        global_label_1_feature_list = []

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

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

            client_i_label_0_feature_list = []
            client_i_label_1_feature_list = []

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                # for batch in client_i_dataloader:
                for batch_id, batch in enumerate(client_i_dataloader):
                    label_0_feature_list = []
                    label_1_feature_list = []

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

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

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

                    if "SENT_CLF" in param_dict["task"]:
                        del activated_preds, logits
                    gc.collect()
                    torch.cuda.empty_cache()

                    # 添加原型素材
                    features = features.cpu()
                    sent_label_flag = labels.gt(0.5)
                    for idx, feat in enumerate(features):
                        if sent_label_flag[idx]:
                            label_1_feature_list.append(feat)
                            client_i_label_1_feature_list.append(feat)
                        else:
                            label_0_feature_list.append(feat)
                            client_i_label_0_feature_list.append(feat)

                    if "SENT_CLF" in param_dict["task"]:
                        del labels, sent_label_flag
                    gc.collect()
                    torch.cuda.empty_cache()


                    # 计算gap
                    (label_0_feature_gap, label_1_feature_gap) = 0, 0
                    # 全局-本地类原型的差异（保留梯度以影响模型训练）
                    if len(label_0_feature_list) != 0:
                        label_0_prototype = torch.stack(label_0_feature_list, dim=0).mean(dim=0)
                        if len(global_label_0_prototype_list) != 0:
                            label_0_feature_gap = torch.norm((global_label_0_prototype_list[-1].cuda() - label_0_prototype.cuda()),
                                                             p=2)
                    if len(label_1_feature_list) != 0:
                        label_1_prototype = torch.stack(label_1_feature_list, dim=0).mean(dim=0)
                        if len(global_label_1_prototype_list) != 0:
                            label_1_feature_gap = torch.norm((global_label_1_prototype_list[-1].cuda() - label_1_prototype.cuda()),
                                                             p=2)
                    lamda_list = [1, 1]  # FedPro思路
                    gap_list = [label_0_feature_gap, label_1_feature_gap]
                    for index, lamda in enumerate(lamda_list):
                        loss += lamda * gap_list[index]

                    loss.backward()

                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()
                        # 清空梯度
                        model.zero_grad()


                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()

                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 清空梯度
                    model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels

                    gc.collect()
                    torch.cuda.empty_cache()

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # 记录GPU计算开始时间
            gpu_start_time = time.time()
            # 计算客户的 类原型
            if len(client_i_label_0_feature_list) != 0:
                client_i_label_0_prototype = torch.stack(client_i_label_0_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_label_0_feature_list.append(client_i_aggregation_weight * client_i_label_0_prototype)

            if len(client_i_label_1_feature_list) != 0:
                client_i_label_1_prototype = torch.stack(client_i_label_1_feature_list, dim=0).mean(dim=0)
                # 由于在内层循环容易获得权重，所以先对原型做加权，方便后续操作
                global_label_1_feature_list.append(client_i_aggregation_weight * client_i_label_1_prototype)
            # 记录GPU计算结束时间
            gpu_end_time = time.time()
            users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Communicate
        logger.info(f"Communicate: {(iter_t + 1)}")
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Total's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2  * prototype_MB_size} MB")

        # Global operation
        logger.info("Prototype aggregation")
        (global_label_0_prototype, global_label_1_prototype) = 0, 0

        # 前面已经乘过权重了，所以这里只需要加起来即可
        if len(global_label_0_feature_list) != 0:
            for proto in global_label_0_feature_list:
                global_label_0_prototype += proto
            global_label_0_prototype_list.append(global_label_0_prototype)  # 更新全局的各种原型
        if len(global_label_1_feature_list) != 0:
            for proto in global_label_1_feature_list:
                global_label_1_prototype += proto
            global_label_1_prototype_list.append(global_label_1_prototype)  # 更新全局的各种原型


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
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedProto.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * prototype_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
