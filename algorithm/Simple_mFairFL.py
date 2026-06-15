import copy
import os
import gc
import random
import time
import torch
import math
import numpy as np
import traceback

from tool.logger import *
from tool.utils import get_parameters, set_parameters, cos_sim, FL_fairness_and_accuracy_test_4_IMG_CLF, get_HM_by_two_value
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from hypothesis.generator import LatentGenerator, FigGenerator


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# FedAvg+FedProx的采样方法+可学习的超参数+批更新内加入群组贱的损失差异+
# 可以说是一种参考AAAI2024 mFairFL的baseline弱化版实验
# https://arxiv.org/pdf/2312.05551
def Simple_mFairFL(device,
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


        # Simulate Client Parallel
        for id in idxs_users:
            client_i_aggregation_weight = average_weight[id]

            # Local Initialization
            # 下发模型
            logger.info(f"Client {id} Init Local Model By Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)

            lambda_param = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))

            lambda_param_optimizer = torch.optim.SGD([lambda_param], lr=0.1)

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


                    # 引入了群组损失差异(对齐)限制--群组梯度差异，增强群组公平性
                    '''
                    这个地方可以形式化为我们用Lagrangian approach来构建了一个Constrain
                    具体的数学写法可以参考AAAI 2024的文章 https://arxiv.org/pdf/2312.05551v1 里面的Eq.10到Eq.12
                    实际情况里面的Performance Gap比较小，对整个效果影响没那么显著，而且属于是间接性优化
                    '''
                    group_flag = protecteds.gt(0.5)
                    one_batch_group_1_count = sum(group_flag)
                    one_batch_group_0_count = true_batch_size - sum(group_flag)
                    if (one_batch_group_1_count != 0) and (one_batch_group_0_count != 0):
                        one_batch_group_1_avg_loss = sum(batch_loss[group_flag]) / one_batch_group_1_count
                        one_batch_group_0_avg_loss = (sum(batch_loss) - sum(batch_loss[group_flag])) / one_batch_group_0_count
                        one_batch_group_avg_loss_gap = torch.abs(one_batch_group_0_avg_loss - one_batch_group_1_avg_loss)
                        if float(batch_id) % 50 == 0:
                            logger.info(f"Origin task loss：{loss.item()} ;\n"
                                        f"one_batch_group_avg_loss_gap: {one_batch_group_avg_loss_gap.item()} ;\n"
                                        f"lambda_param: {lambda_param} ;\n"
                                        f"in batch_id:{batch_id} of epoch:{epoch} in Client:{id}.")
                        loss += lambda_param * one_batch_group_avg_loss_gap

                    loss.backward()
                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()

                        if (one_batch_group_1_count != 0) and (one_batch_group_0_count != 0):
                            grad_lambda = torch.autograd.grad(
                                outputs=lambda_param * one_batch_group_avg_loss_gap,
                                inputs=lambda_param,
                                create_graph=False,
                                retain_graph=False,
                                only_inputs=True
                            )[0]
                            lambda_param.grad = -grad_lambda
                            lambda_param_optimizer.step()

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
                    torch.cuda.empty_cache()
                    # break # 这里记得删除哦

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



            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)

        # Global operation
        # 更新delta

        logger.info(f"Communication Round {(iter_t + 1)} "
                    f"Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # 先读取正常客户的参数
        theta_list = []
        aggregation_weights = []

        for id in idxs_users:

            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            theta_list.append(get_parameters(selected_model))
            aggregation_weights.append(client_datasets_size_list[id]) # 这个地方只需要读取客户的数据量，不用除以总量！
            del selected_model
            gc.collect()
        try:
            if (len(aggregation_weights) != 0) and (sum(aggregation_weights) != 0):
                logger.info("Parameter aggregation")
                theta_list = np.array(theta_list, dtype=object)
                # FedAvg新版论文的聚合权重是数据占比
                # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
                # 如一个weights = [w1, w2, w3, w4]
                # 那么结果就是(theta1 * w1 + theta2 * w2 + theta3 * w3 + theta4 * w4)/ sum(w1+w2+w3+w4)
                theta_avg = np.average(theta_list, axis=0, weights=aggregation_weights).tolist()
                # FedAvg旧版论文的聚合权重是平均
                # theta_avg = np.mean(theta_list, 0).tolist()

                logger.info("Update Global Model with aggregated parameters")
                set_parameters(global_model, theta_avg)
                del theta_list
                gc.collect()
        except Exception as e:
            logger.error(f"Something error happen in loading the Parameter aggregation! Skip! The info: {e}")

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

        # 保存检查点（按 checkpoint_save_freq 间隔）
        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time
            )

            # 清理旧检查点，保留最近 N 个
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_Simple_mFairFL.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
