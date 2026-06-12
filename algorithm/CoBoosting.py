# https://openreview.net/pdf?id=tm8s3696Ox
# ENHANCING ONE-SHOT FEDERATED LEARNINGTHROUGH DATA AND ENSEMBLE CO-BOOSTING

# To ensure fair comparisons, we omit comparisons with methodsthat require the use of auxiliary public datasets, such as Li et al. (2021), or the modification of thelocal training phases of each client, as seen in Diao et al. (2023) and Heinbaugh et al. (2023).
# 没有说明自己用哪个generator，代码里面写了一个，但是不适用文本数据，所以先模拟
# 需要在Server准备一份数据集进行后续的训练，对服务端数据集的依赖
# 超参数epsilon = 8/255, T=200, T_G=30, synthesis_batch_size=128, lr_g=1e-3, kd_lr=0.01, client=all, local_epoch = 100, local_lr = 0.01

import math
import copy
import os
import gc
import time
import torch
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from hypothesis.generator import LatentGenerator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def Co_Boosting(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):
    # Pytorch日志型工具
    torch.autograd.set_detect_anomaly(True)
    accumulation_steps = int(256 / param_dict['batch_size'])
    # 客户端数目
    n = param_dict['num_clients_K']
    # CO_BOOSTING超参数
    epsilon = 8 / 255
    emb_dim = 768
    beta = 1
    miu = 0.1 / n
    # synthesis_batch_size = int(param_dict['batch_size']) # 容易爆24g显存
    synthesis_batch_size = int(param_dict['batch_size']) // n

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    del communication_round_I, FL_fraction, FL_drop_rate, testing_dataloader, testing_dataset_len
    gc.collect()
    torch.cuda.empty_cache()

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
                # logits尺寸 [batch_size, param_dict["le_class"]]
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
                epoch_total_loss += loss.item()
                # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                #     client_datasets_size_list[id] / param_dict['batch_size'])

                del input_ids, attention_mask, labels, features, activated_preds, logits, batch_loss, loss, batch
                gc.collect()
                torch.cuda.empty_cache()

            average_one_sample_loss_in_epoch = float(epoch_total_loss / epoch_total_size)
            logger.info(f"Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            del epoch_total_loss
            gc.collect()
            torch.cuda.empty_cache()

        # Upgrade the local model list
        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        # local_model_list[id] = model.cpu()  # 内存化
        torch.save(model.cpu(), client_model_path)  # 持久化

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # Global operation
    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Communication Round {0} 's Communication Cost: {num_clients_K * 2 * model_MB_size} MB")

    logger.info(f"Create Latent Generator and Noise and Synthetic Samples")
    # 记录GPU计算开始时间
    gpu_start_time = time.time()
    # Generator
    Generator = LatentGenerator(emb_dim)
    # Generator_copy，由于原算法是先更新Generator再用旧的Generator结果来做下一步，所先备份Generator
    Generator_copy = copy.deepcopy(Generator)
    Generator.to(device)
    Generator.train()
    Generator_copy.to(device)
    Generator_copy.train()
    logger.info(f"Create Noise")
    # Noise
    batch_noise_inputs_embeds = torch.rand([synthesis_batch_size, param_dict['max_len'], emb_dim], device='cuda')
    batch_noise_attention_mask = torch.tensor(
        [[1 for i in range(param_dict['max_len'])] for j in range(synthesis_batch_size)], device='cuda')
    batch_noise_token_type_ids = torch.tensor(
        [[0 for i in range(param_dict['max_len'])] for j in range(synthesis_batch_size)], device='cuda')
    batch_noise_label = torch.round(torch.rand(synthesis_batch_size, device='cuda')).long()

    logger.info(f"Training the Generator")
    generator_learning_rate = param_dict['learning_rate'] / 10 # 原论文的生成器更新率就是本地训练的大约10分之1
    generator_criterion = torch.nn.CrossEntropyLoss(reduction='none') # 后面要乘以权重，所以这里不采用reduction='mean'
    # 按照原论文的算法第7行到第10行，如果Generator只练1个epoch，实际上是不用练的
    # for epoch in range(int(math.floor(algorithm_epoch_T * 0.3))):  # 原论文训练的次数就是本地训练的大约3分之1，大约就是1次
    if int(math.floor(algorithm_epoch_T * 0.3)) == 0:
        logger.info(f"Create Synthetic Samples")
        # Synthetic Samples
        batch_synthetic_samples = Generator(batch_noise_inputs_embeds).to(device)
        # 下面是关于使用clone和detach的尝试，全部都不太行，建议废弃
        # 先复制一份Synthetic Samples，后面Generator的参数更新以后要用原始的样本，容易导致运算图不一致报错
        # clone不共享内存地址，但新tensor的梯度会叠加在源tensor上,detach()函数返回与调用对象相关的一个tensor，此新与源tensor共享内存，但其requires_grad为False，并且不包含源tensor的计算图信息
        # 可以简单理解为clone是深拷贝，detach是浅拷贝
        # batch_synthetic_samples_copy = batch_synthetic_samples.clone()
        # 复制的Synthetic Samples需要开梯度计算图，后面要取梯度
        # batch_synthetic_samples_copy.requires_grad=True # 如果使用clone，则需要开这一行
        # batch_synthetic_samples_copy.retain_grad()

        # 这种写法内存地址绝对是不一样的
        # batch_synthetic_samples_copy = Generator(batch_noise_inputs_embeds).to(device)
        # 这种写法不会有BUG
        batch_synthetic_samples_copy = Generator_copy(batch_noise_inputs_embeds).to(device)
    else:
        generator_optimizer = torch.optim.SGD(Generator.parameters(), generator_learning_rate, weight_decay=1e-4, momentum=0.9)
        for epoch in range(int(math.floor(algorithm_epoch_T * 0.3))):
            logger.info(f"Create Synthetic Samples")
            # Synthetic Samples
            batch_synthetic_samples = Generator(batch_noise_inputs_embeds).to(device)
            # 下面是关于使用clone和detach的尝试，全部都不太行，建议废弃
            # 先复制一份Synthetic Samples，后面Generator的参数更新以后要用原始的样本，容易导致运算图不一致报错
            # clone不共享内存地址，但新tensor的梯度会叠加在源tensor上,detach()函数返回与调用对象相关的一个tensor，此新与源tensor共享内存，但其requires_grad为False，并且不包含源tensor的计算图信息
            # 可以简单理解为clone是深拷贝，detach是浅拷贝
            # batch_synthetic_samples_copy = batch_synthetic_samples.clone()
            # 复制的Synthetic Samples需要开梯度计算图，后面要取梯度
            # batch_synthetic_samples_copy.requires_grad=True # 如果使用clone，则需要开这一行
            # batch_synthetic_samples_copy.retain_grad()

            # 这种写法内存地址绝对是不一样的
            # batch_synthetic_samples_copy = Generator(batch_noise_inputs_embeds).to(device)
            # 这种写法不会有BUG
            batch_synthetic_samples_copy = Generator_copy(batch_noise_inputs_embeds).to(device)
            with torch.no_grad(): # 从整体来看，这个环节各个客户端运算的计算图是不用保留的，只需要保留全局模型运算的计算图
                logger.info(f"Using Client Models to Inference")
                client_logit_list = []
                for id in idxs_users:
                    client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
                    client_model = torch.load(client_model_path, weights_only=False)
                    client_model = client_model.to(device)
                    client_model.eval()
                    # client_logit尺寸[batch_size, category]
                    _, client_logit = client_model.latent_forward(batch_synthetic_samples, batch_noise_attention_mask, batch_noise_token_type_ids)
                    client_logit_list.append(client_logit.cpu())

                    del _, client_logit, client_model
                    gc.collect()
                    torch.cuda.empty_cache()

                # client_batch_logit尺寸[num_clients_K, batch_size, category]
                client_batch_logit = torch.stack(client_logit_list).to(device)
                del client_logit_list
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(f"Create the Ensemble parameter and Ensemble Result")
            # ensembled_batch_logit尺寸[batch_size, category]
            ensembled_batch_logit = client_batch_logit.mean(dim=0).to(device)  # 原论文提供的代码就是平均各个client的结果处理
            difficulty_of_batch_synthetic_samples = 0
            for synthetic_index, synthetic_label in enumerate(batch_noise_label):
                ensembled_logit = ensembled_batch_logit[synthetic_index]
                ensembled_logit_pr = ensembled_logit[synthetic_label]
                difficulty_of_batch_synthetic_samples += 1 - ensembled_logit_pr
            L_H = (difficulty_of_batch_synthetic_samples * generator_criterion(ensembled_batch_logit, batch_noise_label)).mean()

            logger.info(f"Using Global Model to Inference")
            global_model.to(device)
            global_model.eval()
            _, global_logit = global_model.latent_forward(batch_synthetic_samples, batch_noise_attention_mask, batch_noise_token_type_ids)
            L_A = -torch.nn.functional.kl_div(ensembled_batch_logit, global_logit) # kl_div函数默认reduction是mean，不用做除法
            # KL散度容易出现NaN,加入防溢出处理
            L_A = torch.nan_to_num(L_A, 0)

            generator_criterion_loss = L_A + (beta * L_H)
            generator_criterion_loss.backward()
            generator_optimizer.step()
            del _, global_logit, L_A, L_H, generator_criterion_loss, client_batch_logit
            gc.collect()
            # 清空计算图
            torch.cuda.empty_cache()


    logger.info(f"Hard Samples Construction")
    gradients_list = []
    # 优先构建random参数，产生计算图
    # batch_random_w = torch.FloatTensor(*client_logit.shape).uniform_(-1., 1.).to(device)
    tmp = torch.rand(synthesis_batch_size, param_dict["le_class"])
    batch_random_w = torch.FloatTensor(*tmp.shape).uniform_(-1., 1.).to(device)
    batch_random_w.requires_grad = True
    # client_logit_list尺寸[idxs_users, category]
    random_w_sign_list = []
    # client_logit_list尺寸[idxs_users, batch_size, category]
    client_logit_list = []
    for index, id in enumerate(idxs_users):
        logger.info(f"id: {id}")
        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        client_model = torch.load(client_model_path, weights_only=False).to(device)
        client_model.eval()
        for param in client_model.parameters():
            param.requires_grad = False
        # 一定要加这行，不然后面会发现取不到batch_synthetic_samples_copy的梯度
        batch_synthetic_samples_copy.retain_grad()
        # client_logit尺寸[batch_size, category]
        _, client_logit = client_model.latent_forward(batch_synthetic_samples_copy, batch_noise_attention_mask, batch_noise_token_type_ids)
        client_logit_list.append(client_logit)
        # 优先计算损失和梯度，释放计算图
        ensembled_batch_logit = (batch_random_w * client_logit)
        # 一定要加这行，不然后面会发现取不到batch_synthetic_samples_copy的梯度
        ensembled_batch_logit.retain_grad()
        tmp_loss = generator_criterion(ensembled_batch_logit, batch_noise_label).mean()
        # 在最后一个客户端执行之前，要保留Generator_copy的计算图
        if index != len(idxs_users) - 1:
            tmp_loss.backward(retain_graph=True)
        else:
            tmp_loss.backward()

        tmp_gradients = batch_synthetic_samples_copy.grad
        # print(f"tmp_gradients: {tmp_gradients}")
        gradients_list.append(tmp_gradients.cpu())
        random_w_sign = batch_random_w.grad.sign()
        random_w_sign_list.append(random_w_sign.cpu())

        # 释放计算图
        del _, client_logit, ensembled_batch_logit, client_model, tmp_loss, tmp_gradients,  random_w_sign, param
        gc.collect()
        torch.cuda.empty_cache()
        # print("A")
        # print(torch.cuda.memory_summary())

    # print(torch.cuda.memory_summary())

    gradients = torch.stack(gradients_list).mean(dim=0).to(device)
    del tmp, gradients_list, batch_synthetic_samples_copy
    gc.collect()
    torch.cuda.empty_cache()
    # print("B")
    # print(torch.cuda.memory_summary())

    try:
        gradients_L2_norm = torch.norm(gradients, p=2)
        hard_batch_synthetic_samples = batch_synthetic_samples + epsilon * (gradients / gradients_L2_norm)
    except Exception:
        hard_batch_synthetic_samples = batch_synthetic_samples

    del gradients, gradients_L2_norm, batch_synthetic_samples
    gc.collect()
    torch.cuda.empty_cache()
    # print("C")
    # print(torch.cuda.memory_summary())

    logger.info(f"Obtain a better ensemble")
    # 计算均值和标准差并归一化
    mean_vals = batch_random_w.mean(dim=0)
    std_vals = batch_random_w.std(dim=0)
    tmp_batch_random_w = batch_random_w - miu * torch.stack(random_w_sign_list).mean(dim=0).to(device)
    normed_random_w = (tmp_batch_random_w - mean_vals) / std_vals
    updated_ensembled_batch_logit = (normed_random_w * torch.stack(client_logit_list).mean(dim=0).to(device)).sum(dim=0)
    del mean_vals, std_vals, random_w_sign_list, tmp_batch_random_w, normed_random_w, client_logit_list, batch_noise_inputs_embeds
    gc.collect()
    torch.cuda.empty_cache()
    # print("D")
    # print(torch.cuda.memory_summary())

    del Generator, Generator_copy
    gc.collect()
    torch.cuda.empty_cache()
    # print("E")
    # print(torch.cuda.memory_summary())

    # 更新全局模型
    logger.info(f"Training the Global Model")
    global_model = global_model.to(device)
    global_model.train()
    global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    global_optimizer.set_parameters(list(global_model.named_parameters()))
    _, hard_global_logit = global_model.latent_forward(hard_batch_synthetic_samples, batch_noise_attention_mask, batch_noise_token_type_ids)
    global_loss = torch.nn.functional.kl_div(updated_ensembled_batch_logit, hard_global_logit)
    try:
        global_loss.backward()
        global_optimizer.step()
        global_optimizer.zero_grad()

    except Exception:
        global_optimizer.zero_grad()
    del _, hard_batch_synthetic_samples, hard_global_logit, updated_ensembled_batch_logit, global_loss, global_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    # print("F")
    # print(torch.cuda.memory_summary())

    del batch_noise_attention_mask, batch_noise_label, batch_noise_token_type_ids, batch_random_w, client_i_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    # print("G")
    # print(torch.cuda.memory_summary())

    # 记录GPU计算结束时间
    gpu_end_time = time.time()
    total_gpu_seconds += (gpu_end_time - gpu_start_time)

    # 当前消耗的总GPU秒，平均GPU秒
    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(
        f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_CoBoosting.pt")
    torch.save(global_model, save_path)

    total_communication_cost = num_clients_K * model_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost