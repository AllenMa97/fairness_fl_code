# https://openreview.net/pdf?id=YjvTJlcb8T
# FedFisher: Leveraging Fisher Information for One-Shot Federated Learning (AISTATS 2024)
# 该方法进行了理论误差分析，每个客户本地训练的同时需要计算一个Fisher矩阵并提交到服务器，服务器根据各个Fisher矩阵对全局模型进行二次训练，计算fisher矩阵的过程代价很高且不稳定

import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from tool.utils import get_parameters, set_parameters


def _compute_diagonal_fisher(model, dataloader, device, param_dict):
    model.eval()
    fisher_diagonals = {}
    for name, param in model.named_parameters():
        fisher_diagonals[name] = torch.zeros_like(param.data)

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='mean')

    num_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if num_samples >= 256:
                break

            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"].to(device)
                labels = batch["labels"].to(device)
                preds, _ = model(imgs)
                loss = criterion(preds[:, 0], labels.float())
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"].to(device)
                labels = batch["labels"].to(device)
                if "ANN" in str(type(model)):
                    pred, _ = model(X)
                elif "LogisticRegression" in str(type(model)):
                    pred = model(X)
                else:
                    pred = model(X)
                loss = criterion(pred[:, 0], labels.float())

            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_diagonals[name] += param.grad.data ** 2

            num_samples += labels.size(0)
            model.zero_grad()

    for name in fisher_diagonals:
        fisher_diagonals[name] /= max(num_samples, 1)

    model.zero_grad()
    return fisher_diagonals


def FedFisher(device,
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

    client_fisher_list = []

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

                gc.collect()

            if (batch_id + 1) % accumulation_steps != 0:
                optimizer.step()
                model.zero_grad()

            average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
            logger.info(f"Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

        logger.info(f"Client {id}: Computing Diagonal Fisher Information Matrix")
        fisher_diag = _compute_diagonal_fisher(model, client_i_dataloader, device, param_dict)
        client_fisher_list.append(fisher_diag)

        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        torch.save(model.cpu(), client_model_path)

        del model, optimizer, fisher_diag
        gc.collect()
        torch.cuda.empty_cache()

    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Communication Round 0 's Communication Cost: {num_clients_K * 2 * model_MB_size} MB")

    logger.info("FedFisher Server: Fisher-Weighted Aggregation and Fine-tuning")
    gpu_start_time = time.time()

    theta_list = []
    for id in idxs_users:
        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        selected_model = torch.load(client_model_path)
        theta_list.append(get_parameters(selected_model))
        del selected_model
        gc.collect()

    theta_list = np.array(theta_list, dtype=object)
    theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()

    logger.info("Update Global Model with weighted average")
    set_parameters(global_model, theta_avg)

    logger.info("FedFisher Server: Fine-tuning Global Model with Fisher-weighted Regularization")
    global_model = global_model.to(device)
    global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                         learning_rate=param_dict['learning_rate'] * 0.1, max_grad_norm=0)
    global_optimizer.set_parameters(list(global_model.named_parameters()))

    fisher_finetune_steps = 50
    fisher_lambda = 0.01

    for ft_step in range(fisher_finetune_steps):
        if ft_step % 10 == 0:
            logger.info(f"Fisher fine-tuning step: {ft_step + 1}/{fisher_finetune_steps}")

        for id in idxs_users:
            client_i_dataloader = training_dataloaders[id]

            for batch_id, batch in enumerate(client_i_dataloader):
                if "SENT_CLF" in param_dict["task"]:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)

                labels = batch["labels"].to(device)

                if "SENT_CLF" in param_dict["task"]:
                    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
                    features, logits = global_model(input_ids=input_ids, attention_mask=attention_mask)
                    activated_preds = logits
                    batch_loss = criterion(activated_preds, labels)
                elif "IMG_CLF" in param_dict["task"]:
                    criterion = torch.nn.BCELoss(reduction='none').to(device)
                    preds, features = global_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())
                elif "Tabular_CLF" in param_dict["task"]:
                    criterion = torch.nn.BCELoss(reduction='none').to(device)
                    if "ANN" in str(type(global_model)):
                        local_prediction, features = global_model(X)
                    elif "LogisticRegression" in str(type(global_model)):
                        local_prediction = global_model(X)
                    else:
                        local_prediction = global_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                loss = torch.mean(batch_loss)

                fisher_reg = 0
                for name, param in global_model.named_parameters():
                    if name in client_fisher_list[id]:
                        fisher_reg += (client_fisher_list[id][name].to(device) * (param ** 2)).sum()

                loss = loss + fisher_lambda * fisher_reg
                loss.backward()
                global_optimizer.step()
                global_model.zero_grad()
                global_optimizer.zero_grad()

                del batch_loss, loss, fisher_reg
                gc.collect()
                torch.cuda.empty_cache()

            break

    gpu_end_time = time.time()
    total_gpu_seconds += (gpu_end_time - gpu_start_time)

    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedFisher.pt")
    torch.save(global_model, save_path)

    total_communication_cost = num_clients_K * model_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost
