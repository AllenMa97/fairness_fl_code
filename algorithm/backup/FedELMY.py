# https://arxiv.org/pdf/2404.12130v1
# FedELMY: One-Shot Sequential Federated Learning for Non-IID Data by Enhancing Local Model Diversity (ACM MM 2024)
# 类似FedDC的思想，将模型在客户端之间序列性传输，但是不能保证客户端之间的可信程度，破坏了隐私；而且通信成本高

import copy
import os
import gc
import time
import torch
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from tool.utils import get_parameters, set_parameters


def _cosine_distance(params1, params2):
    flat1 = torch.cat([p.view(-1) for p in params1])
    flat2 = torch.cat([p.view(-1) for p in params2])
    cos_sim = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
    return 1 - cos_sim.item()


def FedELMY(device,
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

    pool_size = param_dict.get('pool_size', 3)
    diversity_threshold = param_dict.get('diversity_threshold', 0.1)

    idxs_users = [i for i in range(num_clients_K)]
    logger.info(f"FedELMY: Sequential training with model pool, clients order: {idxs_users}")

    model_pool = {id: [] for id in idxs_users}

    for seq_idx, id in enumerate(idxs_users):
        logger.info(f"Sequential Step {seq_idx + 1}/{num_clients_K}: Training Client {id}")

        if seq_idx == 0:
            current_model = copy.deepcopy(global_model)
        else:
            prev_id = idxs_users[seq_idx - 1]
            best_model = None
            best_dist = -1
            for pool_model in model_pool[prev_id]:
                dist = _cosine_distance(
                    list(current_model.parameters()),
                    list(pool_model.parameters())
                )
                if dist > best_dist:
                    best_dist = dist
                    best_model = pool_model
            if best_model is not None and best_dist > diversity_threshold:
                current_model = copy.deepcopy(best_model)
                logger.info(f"Client {id} received diverse model from pool (dist={best_dist:.4f})")
            else:
                logger.info(f"Client {id} uses current sequential model (no diverse model found)")

        current_model.train()
        current_model.to(device)
        optimizer = BERTCLF_Optimizer(
            method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        optimizer.set_parameters(list(current_model.named_parameters()))
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
                    features, logits = current_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    activated_preds = logits
                    _, preds = torch.max(activated_preds, dim=1)
                    batch_loss = criterion(activated_preds, labels)
                elif "IMG_CLF" in param_dict["task"]:
                    criterion = torch.nn.BCELoss(reduction='none').to(device)
                    preds, features = current_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())
                elif "Tabular_CLF" in param_dict["task"]:
                    criterion = torch.nn.BCELoss(reduction='none').to(device)
                    if "ANN" in str(type(current_model)):
                        local_prediction, features = current_model(X)
                    elif "LogisticRegression" in str(type(current_model)):
                        local_prediction = current_model(X)
                    else:
                        local_prediction = current_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                loss = torch.sum(batch_loss) / true_batch_size
                loss.backward()

                if (batch_id + 1) % accumulation_steps == 0:
                    optimizer.step()
                    current_model.zero_grad()

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
                current_model.zero_grad()

            average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
            logger.info(f"Sequential Step: {seq_idx + 1}; Client: {id} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            if len(model_pool[id]) < pool_size:
                model_pool[id].append(copy.deepcopy(current_model).cpu())

        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        torch.save(current_model.cpu(), client_model_path)

        del current_model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Communication Cost: {(num_clients_K - 1) * 2 * model_MB_size} MB (sequential)")

    logger.info("FedELMY: Aggregate Final Model from Sequential Chain")
    gpu_start_time = time.time()

    theta_list = []
    aggregation_weights = []
    for id in idxs_users:
        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        selected_model = torch.load(client_model_path)
        theta_list.append(get_parameters(selected_model))
        aggregation_weights.append(client_datasets_size_list[id])
        del selected_model
        gc.collect()

    theta_list = np.array(theta_list, dtype=object)
    theta_avg = np.average(theta_list, axis=0, weights=aggregation_weights).tolist()

    logger.info("Update Global Model")
    set_parameters(global_model, theta_avg)

    gpu_end_time = time.time()
    total_gpu_seconds += (gpu_end_time - gpu_start_time)

    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedELMY.pt")
    torch.save(global_model, save_path)

    total_communication_cost = (num_clients_K - 1) * model_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost
