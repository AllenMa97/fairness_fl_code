# https://arxiv.org/abs/2112.12371v2
# DENSE: Data-Free One-Shot Federated Learning (NeurIPS 2022)
# 核心机制依赖 BN 层的 running statistics 来生成合成数据，仅适用于含 BN 层的模型
# 对模型结构存在依赖，需要有BN层才能用；而且需要练一个GAN，十分不稳定导致昂贵

import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from hypothesis.generator import FigGenerator


def DENSE(device,
          global_model,
          algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
          training_dataloaders,
          training_dataset,
          client_dataset_list,
          param_dict,
          testing_dataloader,
          testing_dataset_len
          ):

    if "IMG_CLF" not in param_dict["task"]:
        raise ValueError(
            f"DENSE requires BatchNorm layers in the model architecture, "
            f"which is only available in IMG_CLF (CNN with ConditionalBatchNorm). "
            f"Current task: {param_dict['task']}. "
            f"DENSE does not support SENT_CLF (BERT has no BN) or Tabular_CLF (ANN/LogReg have no BN)."
        )

    accumulation_steps = max(1, int(256 / param_dict['batch_size']))

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    del communication_round_I, FL_fraction, FL_drop_rate
    gc.collect()

    basic_path = param_dict['model_path']

    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)

    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)

    idxs_users = [i for i in range(num_clients_K)]
    logger.info(f"Communication Round: 0; Select clients: {idxs_users}; Start Local Training!")

    for id in idxs_users:
        logger.info("Copy From Global Model")
        model = copy.deepcopy(global_model)
        model.train()
        model.to(device)
        optimizer = BERTCLF_Optimizer(
            method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        optimizer.set_parameters(list(model.named_parameters()))
        client_i_dataloader = training_dataloaders[id]

        for epoch in range(algorithm_epoch_T):
            epoch_total_loss = 0
            epoch_total_size = 0

            for batch_id, batch in enumerate(client_i_dataloader):
                imgs = batch["img"].to(device)
                labels = batch["labels"].to(device)
                true_batch_size = labels.size()[0]
                epoch_total_size += true_batch_size

                gpu_start_time = time.time()

                criterion = torch.nn.BCELoss(reduction='none').to(device)
                preds, features = model(imgs)
                batch_loss = criterion(preds[:, 0], labels.float())

                loss = torch.sum(batch_loss) / true_batch_size
                loss.backward()

                if (batch_id + 1) % accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()

                gpu_end_time = time.time()
                users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)
                epoch_total_loss += loss

                del imgs, labels
                gc.collect()

            if (batch_id + 1) % accumulation_steps != 0:
                optimizer.step()
                model.zero_grad()

            average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
            logger.info(f"Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        torch.save(model.cpu(), client_model_path)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Communication Round 0 's Communication Cost: {num_clients_K * 2 * model_MB_size} MB")

    gpu_start_time = time.time()

    logger.info("DENSE Stage 1: Data Generation - Training Generator with Ensemble Models")
    emb_dim = 100
    Generator = FigGenerator(nz=emb_dim, ngf=64, img_size=64, nc=3).to(device)

    generator_optimizer = torch.optim.Adam(Generator.parameters(), lr=param_dict['learning_rate'])
    generator_steps = 200
    batch_size = param_dict['batch_size']

    for g_step in range(generator_steps):
        if g_step % 50 == 0:
            logger.info(f"Generator training step: {g_step + 1}/{generator_steps}")

        noise = torch.rand([batch_size, emb_dim], device=device)
        synthetic_data = Generator(noise)
        pseudo_labels = torch.randint(0, 2, (batch_size,), device=device).float()

        with torch.no_grad():
            client_logit_list = []
            for id in idxs_users:
                client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
                client_model = torch.load(client_model_path)
                client_model.eval()
                client_model = client_model.to(device)

                client_logit, _ = client_model(synthetic_data)
                client_logit_list.append(client_logit)

                del client_model
                gc.collect()
                torch.cuda.empty_cache()

            ensembled_logit = torch.stack(client_logit_list).mean(dim=0)

        criterion_gen = torch.nn.BCELoss()
        gen_loss = criterion_gen(ensembled_logit[:, 0], pseudo_labels)

        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()
        Generator.zero_grad()

        del noise, synthetic_data, pseudo_labels, ensembled_logit, gen_loss
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("DENSE Stage 2: Model Distillation - Distill Ensemble Knowledge to Global Model")
    global_model = global_model.to(device)
    global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                         learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    global_optimizer.set_parameters(list(global_model.named_parameters()))

    distillation_steps = 200
    global_criterion = torch.nn.MSELoss()

    for d_step in range(distillation_steps):
        if d_step % 50 == 0:
            logger.info(f"Distillation step: {d_step + 1}/{distillation_steps}")

        with torch.no_grad():
            noise = torch.rand([batch_size, emb_dim], device=device)
            synthetic_data = Generator(noise)

            client_logit_list = []
            for id in idxs_users:
                client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
                client_model = torch.load(client_model_path)
                client_model.eval()
                client_model = client_model.to(device)

                client_logit, _ = client_model(synthetic_data)
                client_logit_list.append(client_logit)

                del client_model
                gc.collect()
                torch.cuda.empty_cache()

            ensembled_logit = torch.stack(client_logit_list).mean(dim=0)

        global_logit, _ = global_model(synthetic_data)

        global_loss = global_criterion(ensembled_logit, global_logit)
        global_loss.backward()
        global_optimizer.step()
        global_model.zero_grad()
        global_optimizer.zero_grad()

        del noise, synthetic_data, ensembled_logit, global_logit, global_loss
        gc.collect()
        torch.cuda.empty_cache()

    gpu_end_time = time.time()
    total_gpu_seconds += (gpu_end_time - gpu_start_time)

    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_DENSE.pt")
    torch.save(global_model, save_path)

    total_communication_cost = num_clients_K * model_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost
