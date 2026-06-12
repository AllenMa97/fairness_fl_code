# https://arxiv.org/pdf/2110.15545

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


def weighted_loss(criterion, logits, targets, weights, mean=True):
    acc_loss = criterion(logits, targets)
    if mean:
        weights_sum = weights.sum().item()
        acc_loss = torch.sum(acc_loss * weights / weights_sum)
    else:
        acc_loss = torch.sum(acc_loss * weights)
    return acc_loss


def weighted_average_weights(w, nc, n):
    w_avg = copy.deepcopy(w[0])
    for i in range(0, len(w)):
        for key in w_avg.keys():
            try:
                w_avg[key] += w[i][key] * nc[i]
            except Exception:
                pass

    for key in w_avg.keys():
        w_avg[key] = torch.div(w_avg[key], n)
    return w_avg


def get_logits_from_logistic(p):
    logits = torch.log(p / (1 - p))
    return logits


def FedFB_style_inference(param_dict, device, model, inference_dataloader, bits=False, truem_yz=None):
    """
    Returns the inference accuracy,
                            loss,
                            N(sensitive group, pos),
                            N(non-sensitive group, pos),
                            N(sensitive group),
                            N(non-sensitive group),
                            acc_loss,
                            fair_loss
    """

    model.eval()
    model = model.to(device)
    loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='sum').to(device)

    for y in [0, 1]:
        for z in range(2):
            loss_yz[(y, z)] = 0
            n_yz[(y, z)] = 0
            m_yz[(y, z)] = 0

    with torch.no_grad():
        for batch in inference_dataloader:
            try:
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

                sensitive = batch["protected"].to(device)
                # Inference & Prediction
                if "SENT_CLF" in param_dict["task"]:
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    _, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, pred_labels = torch.max(activated_preds, dim=1)

                elif "IMG_CLF" in param_dict["task"]:
                    # preds尺寸 [batch_size, 1]，里面只有0和1
                    # features尺寸 [batch_size, emb_dim]
                    # tmp_logits尺寸 [batch_size, 1]，里面是预测为1的概率
                    tmp_logits, preds, _ = model(imgs, return_logit=True)
                    # RegularCNN给出来的结果没有经过归一化，所以得手动处理下
                    tmp_logits_min = tmp_logits.min()
                    tmp_logits_max = tmp_logits.max()
                    tmp_logits_range = tmp_logits_max - tmp_logits_min
                    # logits尺寸 [batch_size, 2]
                    logits = torch.tensor([[(tmp_logits_range-item+tmp_logits_min)/tmp_logits_range, (item-tmp_logits_min)/tmp_logits_range] for item in tmp_logits]).to(device)
                    # pred_labels尺寸 [batch_size, ]，里面只有0和1
                    pred_labels = preds.squeeze(1)

                elif "Tabular_CLF" in param_dict["task"]:
                    # local_prediction尺寸 [batch_size, 1]，里面是预测为1的概率
                    if "ANN" in str(type(model)):
                        local_prediction, features = model(X)
                    elif "LogisticRegression" in str(type(model)):
                        local_prediction = model(X)
                    else:
                        local_prediction = model(X)
                    # pred_labels尺寸 [batch_size, 1]，里面只有True和False
                    pred_labels = (local_prediction >= 0.5).squeeze(1)
                    # 逻辑回归给出来的结果没有经过归一化，有可能inf或者nan，所以得手动处理下
                    tmp_logits_min = local_prediction.min()
                    tmp_logits_max = local_prediction.max()
                    tmp_logits_range = tmp_logits_max - tmp_logits_min
                    # logits尺寸 [batch_size, 2]
                    logits = torch.tensor([[(tmp_logits_range - item + tmp_logits_min) / tmp_logits_range,
                                            (item - tmp_logits_min) / tmp_logits_range] for item in local_prediction]).to(device)


                correct += sum(pred_labels.eq(labels)).item()
                total += len(labels)
                num_batch += 1

                group_boolean_idx = {}

                for yz in n_yz:
                    group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                    n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()
                    m_yz[yz] += torch.sum((labels == yz[0]) & (sensitive == yz[1])).item()

                    try:
                        if group_boolean_idx[yz].sum() != 0:
                            # the objective function have no lagrangian term
                            if "SENT_CLF" in param_dict["task"]:
                                acc_loss = criterion(activated_preds[group_boolean_idx[yz]], labels[group_boolean_idx[yz]])
                            elif "IMG_CLF" in param_dict["task"]:
                                if torch.isnan(preds[group_boolean_idx[yz],0]).any():
                                    acc_loss = torch.tensor(0).to(device)
                                else:
                                    acc_loss = criterion(preds[group_boolean_idx[yz],0], labels[group_boolean_idx[yz]].float())
                            elif "Tabular_CLF" in param_dict["task"]:
                                if torch.isnan(local_prediction[group_boolean_idx[yz],0]).any():
                                    acc_loss = torch.tensor(0).to(device)
                                else:
                                    acc_loss = criterion(local_prediction[group_boolean_idx[yz],0], labels[group_boolean_idx[yz]].float())
                            loss_yz[yz] += acc_loss.item()

                    except Exception as e:
                        logger.info(f"*** Some error Happen during calculating the acc_loss！： {e} ***")


                # logits_1尺寸 [batch_size, 2]
                logits_1 = logits
                logits_0 = 1-logits_1

                fair_loss0 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(),
                                       torch.max(logits_0 - torch.mean(logits_0), dim=1)[1])
                fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0))
                fair_loss1 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(),
                                       torch.max(logits_1 - torch.mean(logits_1), dim=1)[1])
                fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1))
                fair_loss = fair_loss0 + fair_loss1
                if "SENT_CLF" in param_dict["task"]:
                    acc_loss = criterion(activated_preds, labels)
                elif "IMG_CLF" in param_dict["task"]:
                    acc_loss = criterion(preds[:,0], labels.float())
                elif "Tabular_CLF" in param_dict["task"]:
                    acc_loss = criterion(local_prediction[:,0], labels.float())

                batch_loss, batch_acc_loss, batch_fair_loss = acc_loss, acc_loss, fair_loss

                loss, acc_loss, fair_loss = (loss + batch_loss.item(),
                                             acc_loss + batch_acc_loss.item(),
                                             fair_loss + batch_fair_loss)

            except Exception as e:
                continue

        try:
            accuracy = correct / total
        except ZeroDivisionError as zde:
            accuracy = 0
        for z in range(1, 2):
            f_z[z] = - loss_yz[(0, 0)] / (truem_yz[(0, 0)] + truem_yz[(1, 0)])
            + loss_yz[(1, 0)] / (truem_yz[(0, 0)] + truem_yz[(1, 0)])
            + loss_yz[(0, z)] / (truem_yz[(0, z)] + truem_yz[(1, z)])
            - loss_yz[(1, z)] / (truem_yz[(0, z)] + truem_yz[(1, z)])

        # return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, f_z
        # 检查代码发现acc_loss / num_batch 和 fair_loss / num_batch是没有用到的，所以可以直接反0
        return accuracy, loss, n_yz, 0, 0, f_z


def FedFB(device,
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
    # average_weight = np.array([float(i / training_dataset_size) for i in client_datasets_size_list])

    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(num_clients_K) + "Clients")

    # Operation in FedFB
    # the number of samples whose label is y and sensitive attribute is z
    m_yz, lbd = {}, {}
    for y in [0, 1]:
        for z in range(2):
            try:
                m_yz[(y, z)] = sum(([item  == y for item in training_dataset.labels]) and ([item  == z for item in training_dataset.protected]))
            except Exception as e:
                m_yz[(y, z)] = sum(([item  == y for item in training_dataset.labels]) and ([item  == z for item in training_dataset.s1]))
    for y in [0, 1]:
        for z in range(2):
            lbd[(y, z)] = (m_yz[(1, z)] + m_yz[(0, z)]) / len(training_dataset)
    # New Params in FedFB
    alpha = 0.3
    global_nc = []


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
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

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

                nc = 0  # New Param in FedFB

                # 记录GPU计算开始时间
                gpu_start_time = time.time()

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                # for batch_index, batch in enumerate(client_i_dataloader):
                for batch in client_i_dataloader:
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
                    protected = batch["protected"].to(device)

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

                    # Operation in FedFB
                    v = torch.ones(true_batch_size).type(torch.DoubleTensor).to(device)  # New Param in FedFB
                    group_idx = {}  # New Param in FedFB
                    for y, z in lbd:
                        group_idx[(y, z)] = torch.where((labels == y) & (protected == z))[0].to(device)
                        v[group_idx[(y, z)]] = lbd[(y, z)] / (m_yz[(1, z)] + m_yz[(0, z)])
                        nc += v[group_idx[(y, z)]].sum().item()

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
                        # Loss in FedFB
                        batch_loss_sum = weighted_loss(criterion, activated_preds, labels, v, mean=False)

                    elif "IMG_CLF" in param_dict["task"]:
                        # preds尺寸 [batch_size, 1]
                        # features尺寸 [batch_size, emb_dim]
                        preds, features = model(imgs)
                        # Loss in FedFB
                        batch_loss_sum = weighted_loss(criterion, preds[:,0], labels.float(), v, mean=False)

                    elif "Tabular_CLF" in param_dict["task"]:
                        # local_prediction尺寸 [batch_size, 1]
                        if "ANN" in str(type(model)):
                            local_prediction, features = model(X)
                        elif "LogisticRegression" in str(type(model)):
                            local_prediction = model(X)
                        else:
                            local_prediction = model(X)
                        batch_loss_sum = weighted_loss(criterion, local_prediction[:, 0], labels.float(), v, mean=False)

                    if batch_loss_sum.item() != 0:
                        loss = batch_loss_sum / true_batch_size
                        loss.backward()

                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()
                    else:
                        loss = 0

                    # 清空梯度
                    model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels

                    gc.collect()


                # 记录GPU计算结束时间
                gpu_end_time = time.time()

                users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                epoch_total_size = max(epoch_total_size, client_datasets_size_list[id])

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            # Operation in FedFB
            global_nc.append(nc)

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
            theta_list.append(selected_model.state_dict())  # Operation in FedFB
            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        theta_avg = weighted_average_weights(theta_list, global_nc, sum(global_nc))  # Operation in FedFB
        logger.info("Update Global Model")
        global_model.load_state_dict(theta_avg)  # Operation in FedFB
        # 记录GPU计算结束时间
        gpu_end_time = time.time()
        total_gpu_seconds += (gpu_end_time - gpu_start_time)


        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        # Calculate avg training accuracy over all clients at every round
        list_acc = []
        # the number of samples which are assigned to class y and belong to the sensitive group z
        n_yz, f_z = {}, {}
        for z in range(2):
            f_z[z] = 0
            for y in [0, 1]:
                n_yz[(y, z)] = 0
        global_model.eval()
        for i in range(num_clients_K):
            client_i_dataloader = training_dataloaders[i]
            acc, loss, n_yz_c, acc_loss, fair_loss, f_z_c = FedFB_style_inference(param_dict, device, global_model,
                                                                                  client_i_dataloader, False, m_yz)
            list_acc.append(acc)

            for yz in n_yz:
                n_yz[yz] += n_yz_c[yz]

            for z in range(1, 2):
                f_z[z] += f_z_c[z]
                tmp_0 =  m_yz[(0, 0)] + m_yz[(1, 0)]
                if tmp_0 != 0:
                    f_z[z] += m_yz[(0, 0)] / tmp_0
                tmp_1 =  m_yz[(0, z)] + m_yz[(1, z)]
                if tmp_1 != 0:
                    f_z[z] += -m_yz[(0, z)] / tmp_1

        for z in range(2):
            if z == 0:
                lbd[(0, z)] -= alpha / (iter_t + 1) ** .5 * sum([f_z[z] for z in range(1, 2)])
                # lbd[(0, z)] = lbd[(0, z)].item()
                lbd[(0, z)] = max(0, min(lbd[(0, z)], 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset)))
                lbd[(1, z)] = 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset) - lbd[(0, z)]
            else:
                lbd[(0, z)] += alpha / (iter_t + 1) ** .5 * f_z[z]
                # lbd[(0, z)] = lbd[(0, z)].item()
                lbd[(0, z)] = max(0, min(lbd[(0, z)], 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset)))
                lbd[(1, z)] = 2 * (m_yz[(1, 0)] + m_yz[(0, 0)]) / len(training_dataset) - lbd[(0, z)]
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
            try:
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
            except Exception as e:
                logger.info("Exception: {}".format(e))
                logger.info("!!!!!!!!!!!!! Skipping the middle test in communication round {} !!!!!!!!!".format(iter_t + 1))

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedFB.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
