# One-Shot Federated Learning

import copy
import os
import gc
import time
import math
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer


def OneShotFed(device,
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

    emb_dim = 768
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    del communication_round_I, FL_fraction, FL_drop_rate, testing_dataloader, testing_dataset_len
    gc.collect()

    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(num_clients_K) + "Clients")

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
    idxs_users = [i for i in range(num_clients_K)]

    logger.info(f"Communication Round: {0}; Select clients: {idxs_users}; Start Local Training!")

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
            for batch_id, batch in enumerate(client_i_dataloader):
            # for batch in client_i_dataloader:
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

                del input_ids, attention_mask, labels
                gc.collect()

            average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
            logger.info(f"Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

        # Upgrade the local model list
        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        # local_model_list[id] = model.cpu()  # 内存化
        torch.save(model.cpu(), client_model_path)  # 持久化

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Communicate
    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Communication Round {0} 's Communication Cost: {num_clients_K * 2 * model_MB_size} MB")

    # Global operation
    # 记录GPU计算开始时间
    gpu_start_time = time.time()
    with torch.no_grad():
        logger.info(f"Create Synthetic Samples")
        # batch_noise尺寸[batch_size, seq_length, embedding_dim]
        batch_noise_inputs_embeds = torch.rand([param_dict['batch_size'], param_dict['max_len'],emb_dim])
        batch_noise_attention_mask = torch.tensor([[1 for i in range(param_dict['max_len'])] for j in range(param_dict['batch_size'])])
        batch_noise_token_type_ids = torch.tensor([[0 for i in range(param_dict['max_len'])] for j in range(param_dict['batch_size'])])
        batch_noise_inputs_embeds = batch_noise_inputs_embeds.to(device)
        batch_noise_attention_mask = batch_noise_attention_mask.to(device)
        batch_noise_token_type_ids = batch_noise_token_type_ids.to(device)

        logger.info(f"Using Client Models to Inference")
        client_logit_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            client_model = torch.load(client_model_path, weights_only=False)
            client_model.eval()
            client_model = client_model.to(device)
            # client_logit尺寸[batch_size, category]
            _, client_logit = client_model.latent_forward(batch_noise_inputs_embeds, batch_noise_attention_mask, batch_noise_token_type_ids)
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
    _, global_logit = global_model.latent_forward(batch_noise_inputs_embeds, batch_noise_attention_mask, batch_noise_token_type_ids)
    global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    global_optimizer.set_parameters(list(global_model.named_parameters()))

    global_criterion = torch.nn.MSELoss()
    global_loss = global_criterion(ensembled_batch_logit, global_logit)
    global_loss.backward()
    logger.info("Update Global Model")
    global_optimizer.step()
    # 记录GPU计算结束时间
    gpu_end_time = time.time()
    total_gpu_seconds+= (gpu_end_time - gpu_start_time)

    # 当前消耗的总GPU秒，平均GPU秒
    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(
        f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_OSFL.pt")
    torch.save(global_model, save_path)

    total_communication_cost = num_clients_K * model_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost