# Preserving privacy in federated learning with ensemble cross-domain knowledge distillation
# https://arxiv.org/pdf/2209.04599 (AAAI 2022)
# 模型无关方法，需要服务器端有公共数据集（或合成数据）进行知识蒸馏
# 主要解决的是领域迁移问题，用了无标签的公共数据集在服务器进行训练，可能引入新的偏差

import copy
import os
import gc
import time
import torch
import torch.nn as nn
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from hypothesis.generator import LatentGenerator, FigGenerator


class TabularGenerator(nn.Module):
    def __init__(self, nz, output_dim):
        super(TabularGenerator, self).__init__()
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def FedKD(device,
          global_model,
          algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
          training_dataloaders,
          training_dataset,
          client_dataset_list,
          param_dict,
          testing_dataloader,
          testing_dataset_len
          ):

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
                if "SENT_CLF" in param_dict["task"]:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)

                labels = batch["labels"].to(device)
                true_batch_size = labels.size()[0]
                epoch_total_size += true_batch_size

                gpu_start_time = time.time()

                if "SENT_CLF" in param_dict["task"]:
                    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
                    features, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    activated_preds = logits
                    _, preds = torch.max(activated_preds, dim=1)
                    batch_loss = criterion(activated_preds, labels)
                elif "IMG_CLF" in param_dict["task"]:
                    criterion = torch.nn.BCELoss(reduction='none').to(device)
                    preds, features = model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())
                elif "Tabular_CLF" in param_dict["task"]:
                    criterion = torch.nn.BCELoss(reduction='none').to(device)
                    if "ANN" in str(type(model)):
                        local_prediction, features = model(X)
                    elif "LogisticRegression" in str(type(model)):
                        local_prediction = model(X)
                    else:
                        local_prediction = model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                loss = torch.sum(batch_loss) / true_batch_size
                loss.backward()

                if (batch_id + 1) % accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()

                gpu_end_time = time.time()
                users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)
                epoch_total_loss += loss

                if "SENT_CLF" in param_dict["task"]:
                    del input_ids, attention_mask, labels
                elif "IMG_CLF" in param_dict["task"]:
                    del imgs, labels
                elif "Tabular_CLF" in param_dict["task"]:
                    del X, labels

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

    logger.info("FedKD Server: Cross-Domain Knowledge Distillation with Synthetic Public Data")
    gpu_start_time = time.time()

    if "SENT_CLF" in param_dict["task"]:
        emb_dim = 768
        Generator = LatentGenerator(emb_dim).to(device)
    elif "IMG_CLF" in param_dict["task"]:
        emb_dim = 100
        Generator = FigGenerator(nz=emb_dim, ngf=64, img_size=64, nc=3).to(device)
    elif "Tabular_CLF" in param_dict["task"]:
        emb_dim = param_dict.get('nn_input_size', 128)
        Generator = TabularGenerator(nz=100, output_dim=emb_dim).to(device)

    global_model = global_model.to(device)
    global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                         learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    global_optimizer.set_parameters(list(global_model.named_parameters()))

    distillation_steps = 200
    batch_size = param_dict['batch_size']
    temperature = param_dict.get('temperature', 3.0)

    for d_step in range(distillation_steps):
        if d_step % 50 == 0:
            logger.info(f"Distillation step: {d_step + 1}/{distillation_steps}")

        with torch.no_grad():
            if "SENT_CLF" in param_dict["task"]:
                noise = torch.rand([batch_size, param_dict['max_len'], emb_dim], device=device)
                synthetic_data = Generator(noise)
                noise_attention_mask = torch.tensor(
                    [[1 for i in range(param_dict['max_len'])] for j in range(batch_size)], device=device)
                noise_token_type_ids = torch.tensor(
                    [[0 for i in range(param_dict['max_len'])] for j in range(batch_size)], device=device)
            elif "IMG_CLF" in param_dict["task"]:
                noise = torch.rand([batch_size, emb_dim], device=device)
                synthetic_data = Generator(noise)
            elif "Tabular_CLF" in param_dict["task"]:
                noise = torch.rand([batch_size, 100], device=device)
                synthetic_data = Generator(noise)

            client_logit_list = []
            for id in idxs_users:
                client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
                client_model = torch.load(client_model_path)
                client_model.eval()
                client_model = client_model.to(device)

                if "SENT_CLF" in param_dict["task"]:
                    _, client_logit = client_model.latent_forward(synthetic_data, noise_attention_mask,
                                                                   noise_token_type_ids)
                elif "IMG_CLF" in param_dict["task"]:
                    client_logit, _ = client_model(synthetic_data)
                elif "Tabular_CLF" in param_dict["task"]:
                    if "ANN" in str(type(client_model)):
                        client_pred, _ = client_model(synthetic_data)
                    elif "LogisticRegression" in str(type(client_model)):
                        client_pred = client_model(synthetic_data)
                    else:
                        client_pred = client_model(synthetic_data)
                    client_logit = torch.cat([1 - client_pred, client_pred], dim=1)

                client_logit_list.append(client_logit)
                del client_model
                gc.collect()
                torch.cuda.empty_cache()

            ensembled_logit = torch.stack(client_logit_list).mean(dim=0)
            teacher_probs = torch.softmax(ensembled_logit / temperature, dim=1)

        if "SENT_CLF" in param_dict["task"]:
            _, global_logit = global_model.latent_forward(synthetic_data, noise_attention_mask,
                                                          noise_token_type_ids)
            student_log_probs = torch.nn.functional.log_softmax(global_logit / temperature, dim=1)
            kd_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            kd_loss = kd_loss * (temperature ** 2)
        elif "IMG_CLF" in param_dict["task"]:
            global_logit, _ = global_model(synthetic_data)
            global_probs = torch.cat([1 - global_logit, global_logit], dim=1)
            student_log_probs = torch.nn.functional.log_softmax(global_probs / temperature, dim=1)
            kd_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            kd_loss = kd_loss * (temperature ** 2)
        elif "Tabular_CLF" in param_dict["task"]:
            if "ANN" in str(type(global_model)):
                global_pred, _ = global_model(synthetic_data)
            elif "LogisticRegression" in str(type(global_model)):
                global_pred = global_model(synthetic_data)
            else:
                global_pred = global_model(synthetic_data)
            global_probs = torch.cat([1 - global_pred, global_pred], dim=1)
            student_log_probs = torch.nn.functional.log_softmax(global_probs / temperature, dim=1)
            kd_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            kd_loss = kd_loss * (temperature ** 2)

        kd_loss.backward()
        global_optimizer.step()
        global_model.zero_grad()
        global_optimizer.zero_grad()

        del noise, synthetic_data, ensembled_logit, teacher_probs, kd_loss
        gc.collect()
        torch.cuda.empty_cache()

    gpu_end_time = time.time()
    total_gpu_seconds += (gpu_end_time - gpu_start_time)

    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedKD.pt")
    torch.save(global_model, save_path)

    total_communication_cost = num_clients_K * model_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost
