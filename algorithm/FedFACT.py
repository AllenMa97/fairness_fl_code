import copy
import os
import gc
import time
import torch
import torch.nn as nn
import numpy as np
from tool.logger import *
from tool.utils import get_parameters, set_parameters, FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection


def compute_confusion_matrix(preds, labels, protected, num_classes=2, num_groups=2, device='cpu'):
    C = torch.zeros(num_groups, num_classes, num_classes, device=device)
    for g in range(num_groups):
        mask = (protected == g)
        if mask.sum() == 0:
            continue
        g_preds = preds[mask]
        g_labels = labels[mask]
        for c_true in range(num_classes):
            for c_pred in range(num_classes):
                C[g, c_true, c_pred] = ((g_labels == c_true) & (g_preds == c_pred)).sum().float()
    return C


def compute_cost_matrix(dual_lambda, dual_mu, num_classes=2, num_groups=2, device='cpu'):
    cost = torch.ones(num_groups, num_classes, num_classes, device=device)
    for g in range(num_groups):
        for y in range(num_classes):
            for y_hat in range(num_classes):
                if y != y_hat:
                    cost[g, y, y_hat] = 1.0 + dual_lambda[g, y] + dual_mu[g, y]
                else:
                    cost[g, y, y_hat] = 1.0
    return cost


def cost_sensitive_loss(logits, labels, protected, cost_matrix, task, device):
    if "SENT_CLF" in task:
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.size(1)
        batch_cost = torch.zeros(labels.size(0), device=device)
        for i in range(labels.size(0)):
            g = int(protected[i].item())
            y = int(labels[i].item())
            for y_hat in range(num_classes):
                batch_cost[i] += cost_matrix[g, y, y_hat].item() * probs[i, y_hat]
        return batch_cost.mean()
    else:
        if logits.dim() > 1 and logits.size(1) > 1:
            probs = torch.softmax(logits, dim=1)
            num_classes = probs.size(1)
            batch_cost = torch.zeros(labels.size(0), device=device)
            for i in range(labels.size(0)):
                g = int(protected[i].item())
                y = int(labels[i].item())
                for y_hat in range(num_classes):
                    batch_cost[i] += cost_matrix[g, y, y_hat].item() * probs[i, y_hat]
            return batch_cost.mean()
        else:
            probs = torch.sigmoid(logits.squeeze(-1))
            batch_cost = torch.zeros(labels.size(0), device=device)
            for i in range(labels.size(0)):
                g = int(protected[i].item())
                y = int(labels[i].item())
                batch_cost[i] += cost_matrix[g, y, 1].item() * probs[i]
                batch_cost[i] += cost_matrix[g, y, 0].item() * (1 - probs[i])
            return batch_cost.mean()


def update_dual_variables(confusion_matrix, dual_var, eta_d, num_groups=2, num_classes=2):
    for g in range(num_groups):
        for y in range(num_classes):
            group_total = confusion_matrix[g, y, :].sum()
            if group_total > 0:
                pred_rate_y = confusion_matrix[g, y, y] / group_total
            else:
                pred_rate_y = 0.0
            dual_var[g, y] = max(0, dual_var[g, y] + eta_d * (pred_rate_y - 0.5))
    return dual_var


def FedFACT(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0
            ):
    accumulation_steps = int(256 / param_dict['batch_size'])
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    gc.collect()

    basic_path = param_dict['model_path']

    eta_d = param_dict.get('eta_d', 0.1)
    eta_w = param_dict.get('eta_w', 0.1)
    w_init = param_dict.get('w_init', 0.5)
    fairness_level = param_dict.get('fairness_level', 0.05)

    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化
    w_k_list = [w_init for _ in range(num_clients_K)]

    global_dual_lambda = torch.zeros(2, 2, device=device)
    local_dual_mu_list = [torch.zeros(2, 2, device=device) for _ in range(num_clients_K)]

    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')

    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for iter_t in range(start_round, communication_round_I):
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        for id in idxs_users:
            logger.info(f"Client {id} Init Local Model")
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            local_model = torch.load(client_model_path, weights_only=False)
            local_model.train()
            local_model.to(device)

            global_model_copy = copy.deepcopy(global_model)
            global_model_copy.eval()
            global_model_copy.to(device)

            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(local_model.named_parameters()))

            client_i_dataloader = training_dataloaders[id]
            mu_k = local_dual_mu_list[id].to(device)
            w_k = w_k_list[id]

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
                    protected = batch["protected"].to(device)
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

                    gpu_start_time = time.time()

                    cost_matrix = compute_cost_matrix(global_dual_lambda, mu_k, num_classes=2, num_groups=2, device=device)

                    if "SENT_CLF" in param_dict["task"]:
                        features_local, logits_local = local_model(input_ids=input_ids, attention_mask=attention_mask)
                        with torch.no_grad():
                            features_global, logits_global = global_model_copy(input_ids=input_ids, attention_mask=attention_mask)

                        logits_ens = w_k * logits_global + (1 - w_k) * logits_local
                        cs_loss = cost_sensitive_loss(logits_ens, labels, protected, cost_matrix, param_dict["task"], device)
                        ce_loss = criterion(logits_local, labels).mean()
                        loss = ce_loss + cs_loss

                    elif "IMG_CLF" in param_dict["task"]:
                        preds_local, features_local = local_model(imgs)
                        with torch.no_grad():
                            preds_global, features_global = global_model_copy(imgs)

                        preds_ens = w_k * preds_global + (1 - w_k) * preds_local
                        cs_loss = cost_sensitive_loss(preds_ens, labels, protected, cost_matrix, param_dict["task"], device)
                        ce_loss = criterion(preds_local[:, 0], labels.float()).mean()
                        loss = ce_loss + cs_loss

                    elif "Tabular_CLF" in param_dict["task"]:
                        if "ANN" in str(type(local_model)):
                            local_prediction, features_local = local_model(X)
                            with torch.no_grad():
                                if "ANN" in str(type(global_model_copy)):
                                    global_prediction, features_global = global_model_copy(X)
                                elif "LogisticRegression" in str(type(global_model_copy)):
                                    global_prediction = global_model_copy(X)
                                else:
                                    global_prediction = global_model_copy(X)
                        elif "LogisticRegression" in str(type(local_model)):
                            local_prediction = local_model(X)
                            with torch.no_grad():
                                global_prediction = global_model_copy(X)
                        else:
                            local_prediction = local_model(X)
                            with torch.no_grad():
                                global_prediction = global_model_copy(X)

                        pred_ens = w_k * global_prediction + (1 - w_k) * local_prediction
                        cs_loss = cost_sensitive_loss(pred_ens, labels, protected, cost_matrix, param_dict["task"], device)
                        ce_loss = criterion(local_prediction[:, 0], labels.float()).mean()
                        loss = ce_loss + cs_loss

                    loss.backward()

                    if (batch_id + 1) % accumulation_steps == 0:
                        optimizer.step()
                        local_model.zero_grad()

                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    epoch_total_loss += loss.item()

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs

                    gc.collect()

                average_one_sample_loss_in_epoch = epoch_total_loss / max(epoch_total_size, 1)
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            local_model.eval()
            all_preds = []
            all_labels = []
            all_protected = []
            with torch.no_grad():
                for batch in client_i_dataloader:
                    if "SENT_CLF" in param_dict["task"]:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        _, logits = local_model(input_ids=input_ids, attention_mask=attention_mask)
                        preds = logits.argmax(dim=1)
                    elif "IMG_CLF" in param_dict["task"]:
                        imgs = batch["img"].to(device)
                        pred_raw, _ = local_model(imgs)
                        preds = (pred_raw >= 0.5).squeeze(1).long()
                    elif "Tabular_CLF" in param_dict["task"]:
                        X = batch["X"].to(device)
                        if "ANN" in str(type(local_model)):
                            pred_raw, _ = local_model(X)
                        elif "LogisticRegression" in str(type(local_model)):
                            pred_raw = local_model(X)
                        else:
                            pred_raw = local_model(X)
                        preds = (pred_raw >= 0.5).squeeze(1).long()

                    all_preds.append(preds.cpu())
                    all_labels.append(batch["labels"])
                    all_protected.append(batch["protected"])

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_protected = torch.cat(all_protected)

            confusion = compute_confusion_matrix(all_preds, all_labels, all_protected, num_classes=2, num_groups=2, device='cpu')
            mu_k = update_dual_variables(confusion, mu_k.cpu(), eta_d, num_groups=2, num_classes=2)
            local_dual_mu_list[id] = mu_k

            w_k_list[id] = max(0.0, min(1.0, w_k_list[id] + eta_w * (0.5 - w_k_list[id])))

            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            torch.save(local_model.cpu(), client_model_path)

            del local_model, global_model_copy
            gc.collect()

        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        logger.info("Parameter aggregation - Global Model")
        theta_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)
            theta_list.append(get_parameters(selected_model))
            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        set_parameters(global_model, theta_avg)

        logger.info("Parameter aggregation - Dual Variables")
        aggregated_mu = torch.zeros(2, 2)
        total_weight = 0
        for id in idxs_users:
            w = client_datasets_size_list[id]
            aggregated_mu += w * local_dual_mu_list[id]
            total_weight += w
        if total_weight > 0:
            aggregated_mu /= total_weight
        global_dual_lambda = 0.5 * global_dual_lambda + 0.5 * aggregated_mu.to(device)

        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time,
                extra_state={
                    'global_dual_lambda': global_dual_lambda.cpu().tolist(),
                    'local_dual_mu_list': [mu.cpu().tolist() for mu in local_dual_mu_list],
                    'w_k_list': w_k_list
                }
            )
            clean_old_checkpoints(param_dict, keep_latest=5)

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedFACT.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
