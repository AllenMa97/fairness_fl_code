# https://arxiv.org/pdf/2107.00233

import copy
import os
import gc
import time
import torch
import numpy as np
import random
from tool.logger import *
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test
import torch.autograd as autograd


def soft_cross_entropy(pred, soft_targets, reduction='none'):
    logsoftmax = torch.nn.LogSoftmax()
    # soft_targets = torch.nn.functional.one_hot(soft_targets,2)
    soft_targets = torch.tensor([[1-item, item] for item in soft_targets]).to(pred.device)
    if reduction =='none':
        return torch.sum(- soft_targets * logsoftmax(pred), 1)
    if reduction =='sum':
        return torch.sum(torch.sum(- soft_targets * logsoftmax(pred), 1))
    if reduction =='mean':
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def get_mashed_data(client_i_dataloader, global_model, device):
    X_list, Y_list = [],[]
    global_model.to(device)
    with torch.no_grad():
        for batch in client_i_dataloader:
            # input_ids尺寸 [batch_size, max_len]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # labels尺寸 [batch_size]
            labels = batch["labels"].to(device)
            # features尺寸 [batch_size, emb_dim]
            features = global_model.only_PLM_forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            X_list.append(features.cpu())
            Y_list.append(labels.cpu())
            del features, labels

        X_tensors = torch.concatenate(X_list)
        Y_tensors = torch.concatenate(Y_list)

        client_i_X_bar = X_tensors.mean(dim=0).view(1, -1).to(device)
        client_i_Y_bar = Y_tensors.float().mean().view(-1).to(device)
        torch.cuda.empty_cache()

    del X_list, Y_list, global_model, X_tensors, Y_tensors
    gc.collect()
    return client_i_X_bar, client_i_Y_bar

def FedMix(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):

    # 引入的超参数λ
    λ = 0.5

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
    # criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    # 替换了交叉熵函数，使得target不必强行为long，可以为float
    criterion = soft_cross_entropy

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")


    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        logger.info("Get the mashed sample firstly!!")
        # 所有人先把特征平均和标签平均传上来给服务器
        X_bar_list = []
        Y_bar_list = []
        with torch.no_grad():
            for id in range(num_clients_K):
                client_i_dataloader = training_dataloaders[id]
                client_i_X_bar, client_i_Y_bar = get_mashed_data(client_i_dataloader, global_model, device)
                X_bar_list.append(client_i_X_bar)
                Y_bar_list.append(client_i_Y_bar)
        Mash_data_MB_size = sum([p.numel() for p in X_bar_list+Y_bar_list]) * 4 / (1024*1024)


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

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        # Simulate Client Parallel
        for id in idxs_users:
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

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                # for batch_index, batch in enumerate(client_i_dataloader):
                for batch in client_i_dataloader:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # labels尺寸 [batch_size]
                    labels = batch["labels"].to(device)

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

                    # 记录GPU计算开始时间
                    gpu_start_time = time.time()

                    # FedMix 新提出
                    # features尺寸 [batch_size, emb_dim]
                    features = model.only_PLM_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # FedMix 新提出
                    mashed_input = random.choice(X_bar_list)
                    mashed_label = random.choice(Y_bar_list)
                    mashed_labels = mashed_label.expand_as(labels)
                    optimizer.zero_grad()
                    scaled_features = (1-λ) * features
                    scaled_features.requires_grad_()

                    # intermediate_logits [batch_size, category]
                    _, intermediate_logits = model.only_clf_forward(scaled_features)

                    l1 = (1 - λ) * criterion(intermediate_logits, labels, reduction='mean')
                    l2 = λ * criterion(intermediate_logits, mashed_labels, reduction='mean')
                    gradients = autograd.grad(
                        outputs=l1, inputs=scaled_features, create_graph=True, retain_graph=True
                    )[0]
                    l3 = λ * torch.inner(
                        gradients.flatten(start_dim=1), mashed_input.flatten(start_dim=1)
                    )
                    l3 = torch.mean(l3)

                    loss = l1 + l2 + l3
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

            del model
            gc.collect()
            # torch.cuda.empty_cache()

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # Global operation
        logger.info("Parameter aggregation")
        theta_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
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
    save_path = os.path.join(save_dir, f"global_NaiveMix.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    total_communication_cost += communication_round_I * Mash_data_MB_size

    return global_model, total_gpu_seconds, total_communication_cost
