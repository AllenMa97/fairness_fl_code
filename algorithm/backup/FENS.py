# https://arxiv.org/pdf/2411.07182v1
# Revisiting Ensembling in One-Shot Federated Learning (NeurIPS 2024)
# 需要在Server准备一份数据集进行后续的训练，对服务端数据集的依赖

import copy
import os
import gc
import time
import torch
import torch.nn as nn
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import get_parameters, set_parameters, FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value


class AggregatorMLP(nn.Module):
    def __init__(self, num_clients, hidden_dim=64):
        super(AggregatorMLP, self).__init__()
        self.num_clients = num_clients
        self.fc1 = nn.Linear(num_clients * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, client_logits):
        x = client_logits.view(client_logits.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def FENS(device,
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
    logger.info(f"FENS Phase 1: Local Training - Select clients: {idxs_users}")

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

        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        torch.save(model.cpu(), client_model_path)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Phase 1 Communication Cost: {num_clients_K * 2 * model_MB_size} MB")

    logger.info("FENS Phase 2: Training Aggregator Model on Server-side Public Data")
    gpu_start_time = time.time()

    aggregator = AggregatorMLP(num_clients=num_clients_K, hidden_dim=64).to(device)
    aggregator_optimizer = torch.optim.Adam(aggregator.parameters(), lr=param_dict['learning_rate'])

    if "SENT_CLF" in param_dict["task"]:
        aggregator_criterion = torch.nn.CrossEntropyLoss()
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        aggregator_criterion = torch.nn.BCELoss()

    aggregator_rounds = param_dict.get('aggregator_rounds', 50)
    batch_size = param_dict['batch_size']

    for agg_round in range(aggregator_rounds):
        if agg_round % 10 == 0:
            logger.info(f"Aggregator training round: {agg_round + 1}/{aggregator_rounds}")

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

                with torch.no_grad():
                    client_logits_list = []
                    for cid in idxs_users:
                        client_model_path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
                        client_model = torch.load(client_model_path)
                        client_model.eval()
                        client_model = client_model.to(device)

                        if "SENT_CLF" in param_dict["task"]:
                            _, client_logit = client_model(input_ids=input_ids, attention_mask=attention_mask)
                            client_probs = torch.softmax(client_logit, dim=1)
                        elif "IMG_CLF" in param_dict["task"]:
                            client_logit, _ = client_model(imgs)
                            client_probs = torch.cat([1 - client_logit, client_logit], dim=1)
                        elif "Tabular_CLF" in param_dict["task"]:
                            if "ANN" in str(type(client_model)):
                                client_pred, _ = client_model(X)
                            elif "LogisticRegression" in str(type(client_model)):
                                client_pred = client_model(X)
                            else:
                                client_pred = client_model(X)
                            client_probs = torch.cat([1 - client_pred, client_pred], dim=1)

                        client_logits_list.append(client_probs)
                        del client_model
                        gc.collect()
                        torch.cuda.empty_cache()

                    stacked_client_logits = torch.cat(client_logits_list, dim=1)

                agg_output = aggregator(stacked_client_logits)

                if "SENT_CLF" in param_dict["task"]:
                    agg_loss = aggregator_criterion(agg_output, labels)
                elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                    agg_loss = aggregator_criterion(agg_output[:, 1], labels.float())

                aggregator_optimizer.zero_grad()
                agg_loss.backward()
                aggregator_optimizer.step()

                del stacked_client_logits, agg_output, agg_loss
                gc.collect()
                torch.cuda.empty_cache()

            break

    logger.info("FENS Phase 3: Distill Aggregator into Global Model")
    global_model = global_model.to(device)
    global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                         learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    global_optimizer.set_parameters(list(global_model.named_parameters()))

    distill_criterion = torch.nn.MSELoss()
    distill_steps = 100

    for d_step in range(distill_steps):
        if d_step % 20 == 0:
            logger.info(f"Distillation step: {d_step + 1}/{distill_steps}")

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

                with torch.no_grad():
                    client_logits_list = []
                    for cid in idxs_users:
                        client_model_path = os.path.join(basic_path, "client_" + str(cid + 1), 'model.pt')
                        client_model = torch.load(client_model_path)
                        client_model.eval()
                        client_model = client_model.to(device)

                        if "SENT_CLF" in param_dict["task"]:
                            _, client_logit = client_model(input_ids=input_ids, attention_mask=attention_mask)
                            client_probs = torch.softmax(client_logit, dim=1)
                        elif "IMG_CLF" in param_dict["task"]:
                            client_logit, _ = client_model(imgs)
                            client_probs = torch.cat([1 - client_logit, client_logit], dim=1)
                        elif "Tabular_CLF" in param_dict["task"]:
                            if "ANN" in str(type(client_model)):
                                client_pred, _ = client_model(X)
                            elif "LogisticRegression" in str(type(client_model)):
                                client_pred = client_model(X)
                            else:
                                client_pred = client_model(X)
                            client_probs = torch.cat([1 - client_pred, client_pred], dim=1)

                        client_logits_list.append(client_probs)
                        del client_model
                        gc.collect()
                        torch.cuda.empty_cache()

                    stacked_client_logits = torch.cat(client_logits_list, dim=1)
                    teacher_output = aggregator(stacked_client_logits)

                if "SENT_CLF" in param_dict["task"]:
                    _, global_logit = global_model(input_ids=input_ids, attention_mask=attention_mask)
                    global_probs = torch.softmax(global_logit, dim=1)
                elif "IMG_CLF" in param_dict["task"]:
                    global_pred, _ = global_model(imgs)
                    global_probs = torch.cat([1 - global_pred, global_pred], dim=1)
                elif "Tabular_CLF" in param_dict["task"]:
                    if "ANN" in str(type(global_model)):
                        global_pred, _ = global_model(X)
                    elif "LogisticRegression" in str(type(global_model)):
                        global_pred = global_model(X)
                    else:
                        global_pred = global_model(X)
                    global_probs = torch.cat([1 - global_pred, global_pred], dim=1)

                distill_loss = distill_criterion(global_probs, teacher_output)
                distill_loss.backward()
                global_optimizer.step()
                global_model.zero_grad()
                global_optimizer.zero_grad()

                del stacked_client_logits, teacher_output, global_probs, distill_loss
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
    save_path = os.path.join(save_dir, f"global_FENS.pt")
    torch.save(global_model, save_path)

    aggregator_MB_size = sum(p.numel() for p in aggregator.parameters()) * 4 / (1024 * 1024)
    total_communication_cost = num_clients_K * model_MB_size * 2 + aggregator_rounds * num_clients_K * aggregator_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost
