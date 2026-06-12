# https://arxiv.org/abs/2009.07999v3
# Distilled One-Shot Federated Learning

import copy
import os
import gc
import time
import math
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from hypothesis.generator import LatentGenerator

# DOSFL超参数
emb_dim = 768
# Here the distill steps Sd = 5, the distill epochs Ed = 10, the distill batch size Bd = 1, and the starting distill learning rate is η0 = 0.01.
Sd = 5
Ed = 10
# Sd = 1
# Ed = 2
η0 = 0.01
# Clients distill the data for E = 30 epochs for image datasets and E = 50 epochs for text datasets with a batch size of B = 512.
E = 50
# E = 1


def DISTILLDATA(param_dict, model, client_i_dataloader, device):
    # We have α = 0.01, τ = 40, α = 0.01, τ = 10, and α = 0.1, τ = 30 forfederated MNIST, IMDB, and TREC-6 respectively.
    alpha = 0.1
    τ = 10
    # Initialize {(˜xj , y˜j , η˜j )}Sd
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

    for e in range(0,E):
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
            # 只取一个batch
            break
        x_grad = distilled_samples.grad
        distilled_samples.data -= alpha * x_grad
        # learning_rate_grad = distilled_learning_rate.grad
        # distilled_learning_rate -= alpha * learning_rate_grad

        # print("e:",e)

    distilled_learning_rate = optimizer.learning_rate
    return distilled_samples, noise_attention_mask, noise_token_type_ids, distilled_labels, distilled_learning_rate

def DistilledOneShotFed(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):

    del training_dataset, client_dataset_list
    del communication_round_I, FL_fraction, FL_drop_rate, testing_dataloader, testing_dataset_len
    gc.collect()

    # Training process
    logger.info("Local Data Distilation process begin!")

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # Simulate Client Parallel
    idxs_users = [i for i in range(num_clients_K)]

    logger.info(f"Communication Round: {0}; Select clients: {idxs_users}; Start Local Training!")

    distilled_samples_list = []
    noise_attention_mask_list = []
    noise_token_type_ids_list = []
    distilled_labels_list = []
    distilled_learning_rate_list = []
    # Simulate Client Parallel
    for id in idxs_users:
        # Local Initialization
        # 下发模型
        logger.info("Copy From Global Model")
        model = copy.deepcopy(global_model)
        client_i_dataloader = training_dataloaders[id]

        # 记录GPU计算开始时间
        gpu_start_time = time.time()
        # 本地数据蒸馏
        logger.info(f"Client {id} generating local distilled data")
        distilled_samples, noise_attention_mask, noise_token_type_ids, distilled_labels, distilled_learning_rate = DISTILLDATA(param_dict, model, client_i_dataloader, device)
        # 记录GPU计算结束时间
        gpu_end_time = time.time()

        users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

        distilled_samples_list.append(distilled_samples)
        noise_attention_mask_list.append(noise_attention_mask)
        noise_token_type_ids_list.append(noise_token_type_ids)
        distilled_labels_list.append(distilled_labels)
        distilled_learning_rate_list.append(distilled_learning_rate)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    distilled_samples_MB_size = num_clients_K * distilled_samples.numel() * 4 / (1024*2)
    noise_attention_mask_MB_size = num_clients_K * noise_attention_mask.numel() * 4 / (1024*2)
    noise_token_type_ids_MB_size = num_clients_K * noise_token_type_ids.numel() * 4 / (1024*2)
    distilled_labels_MB_size = num_clients_K * distilled_labels.numel() * 4 / (1024*2)
    distilled_learning_rate_MB_size = num_clients_K * sys.getsizeof(distilled_learning_rate) / (1024*1024)

    total_communication_cost = num_clients_K * 1 * model_MB_size + distilled_samples_MB_size + noise_attention_mask_MB_size + noise_token_type_ids_MB_size + distilled_labels_MB_size + distilled_learning_rate_MB_size

    # Communicate
    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Communication Round {0} 's Communication Cost: {total_communication_cost} MB")

    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    # Global operation
    # 记录GPU计算开始时间
    gpu_start_time = time.time()
    # 训练全局模型
    logger.info(f"Update the global model.")
    global_model.to(device)
    global_model.train()
    optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(global_model.named_parameters()))
    for id in idxs_users:
        distilled_samples = distilled_samples_list[id]
        noise_attention_mask = noise_attention_mask_list[id]
        noise_token_type_ids = noise_token_type_ids_list[id]
        distilled_labels = distilled_labels_list[id]
        distilled_learning_rate = distilled_learning_rate_list[id]

        # distilled_learning_rate要每个epoch更新，所以要多次设置
        optimizer._set_rate(learning_rate=distilled_learning_rate)

        # 原论文 Algorithm 1 Line 7-12，更新模型参数
        for i in range(0, Ed):
            for j in range(0, Sd):
                # features尺寸 [batch_size, emb_dim]
                # logits尺寸 [batch_size, category]
                features, logits = global_model.latent_forward(distilled_samples[j].unsqueeze(0),
                                                        noise_attention_mask[j].unsqueeze(0),
                                                        noise_token_type_ids[j].unsqueeze(0))
                # activated_preds = logits.softmax(dim=1)
                activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                _, preds = torch.max(activated_preds, dim=1)
                # batch_loss尺寸 [batch_size]
                batch_loss = criterion(activated_preds, distilled_labels[j].unsqueeze(0))

                loss = torch.sum(batch_loss) / 1
                loss.backward()
                optimizer.step()
                global_model.zero_grad()
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
    save_path = os.path.join(save_dir, f"global_DOSFL.pt")
    torch.save(global_model, save_path)

    return global_model, total_gpu_seconds, total_communication_cost