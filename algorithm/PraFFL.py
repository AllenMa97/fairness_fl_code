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


class HyperNetwork(nn.Module):
    def __init__(self, pref_dim, clf_weight_shape, clf_bias_shape, hidden_dim=256):
        super(HyperNetwork, self).__init__()
        self.pref_dim = pref_dim
        total_output = int(np.prod(clf_weight_shape)) + int(np.prod(clf_bias_shape))
        self.net = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_output)
        )
        self.clf_weight_shape = clf_weight_shape
        self.clf_bias_shape = clf_bias_shape
        self._total_output = total_output

    def forward(self, pref):
        out = self.net(pref)
        w_num = int(np.prod(self.clf_weight_shape))
        weight = out[:w_num].view(self.clf_weight_shape)
        bias = out[w_num:w_num + int(np.prod(self.clf_bias_shape))].view(self.clf_bias_shape)
        return weight, bias


def get_clf_layer_info(model, task):
    if "SENT_CLF" in task:
        clf = model.out
    elif "IMG_CLF" in task:
        clf = model.out_layer
    elif "Tabular_CLF" in task:
        clf = model.out_layer
    else:
        raise ValueError(f"Unknown task: {task}")
    return clf.weight.shape, clf.bias.shape


def apply_clf_weights(model, weight, bias, task):
    if "SENT_CLF" in task:
        model.out.weight.data.copy_(weight)
        model.out.bias.data.copy_(bias)
    elif "IMG_CLF" in task:
        model.out_layer.weight.data.copy_(weight)
        model.out_layer.bias.data.copy_(bias)
    elif "Tabular_CLF" in task:
        model.out_layer.weight.data.copy_(weight)
        model.out_layer.bias.data.copy_(bias)


def compute_fairness_loss(preds, protected, task, device):
    if "SENT_CLF" in task:
        pred_labels = preds.argmax(dim=1).float()
    else:
        if preds.dim() > 1 and preds.size(1) > 1:
            pred_labels = preds.argmax(dim=1).float()
        else:
            pred_labels = (preds >= 0.5).float().squeeze()
            if pred_labels.dim() == 0:
                pred_labels = pred_labels.unsqueeze(0)

    protected = protected.float()
    group_0_mask = (protected == 0)
    group_1_mask = (protected == 1)

    pred_0 = pred_labels[group_0_mask]
    pred_1 = pred_labels[group_1_mask]

    if pred_0.numel() == 0 or pred_1.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    dp_gap = torch.abs(pred_0.mean() - pred_1.mean())
    return dp_gap


def PraFFL(device,
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

    pref_bs = param_dict.get('pref_bs', 8)
    tau_p = param_dict.get('tau_p', 0.5)
    hypernet_lr = param_dict.get('hypernet_lr', 1e-3)
    hypernet_hidden = param_dict.get('hypernet_hidden', 256)

    clf_weight_shape, clf_bias_shape = get_clf_layer_info(global_model, param_dict["task"])
    hypernetwork = HyperNetwork(
        pref_dim=1,
        clf_weight_shape=clf_weight_shape,
        clf_bias_shape=clf_bias_shape,
        hidden_dim=hypernet_hidden
    ).to(device)

    for k in range(param_dict["num_clients_K"]):
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)

    hypernet_path = os.path.join(basic_path, "hypernetwork.pt")
    torch.save(hypernetwork.cpu(), hypernet_path)

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
            logger.info(f"Client {id} Init Local Model By Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)

            local_hypernetwork = copy.deepcopy(hypernetwork).to(device)
            hypernet_optimizer = torch.optim.Adam(local_hypernetwork.parameters(), lr=hypernet_lr)

            client_i_dataloader = training_dataloaders[id]

            for epoch in range(algorithm_epoch_T):
                epoch_total_loss = 0
                epoch_total_size = 0

                dataloader_iter = iter(client_i_dataloader)
                batch_id = 0
                while True:
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        break

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

                    pref_loss = torch.tensor(0.0, device=device)
                    for _ in range(pref_bs):
                        alpha = torch.rand(1, device=device)

                        clf_weight, clf_bias = local_hypernetwork(alpha)
                        apply_clf_weights(model, clf_weight, clf_bias, param_dict["task"])

                        if "SENT_CLF" in param_dict["task"]:
                            features, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                            perf_loss = criterion(logits, labels).mean()
                            fair_loss = compute_fairness_loss(logits, protected, param_dict["task"], device)
                        elif "IMG_CLF" in param_dict["task"]:
                            preds, features = model(imgs)
                            perf_loss = criterion(preds[:, 0], labels.float()).mean()
                            fair_loss = compute_fairness_loss(preds, protected, param_dict["task"], device)
                        elif "Tabular_CLF" in param_dict["task"]:
                            if "ANN" in str(type(model)):
                                local_prediction, features = model(X)
                            elif "LogisticRegression" in str(type(model)):
                                local_prediction = model(X)
                            else:
                                local_prediction = model(X)
                            perf_loss = criterion(local_prediction[:, 0], labels.float()).mean()
                            fair_loss = compute_fairness_loss(local_prediction, protected, param_dict["task"], device)

                        tche_loss = torch.max(alpha * perf_loss, (1 - alpha) * fair_loss)
                        pref_loss = pref_loss + tche_loss

                    pref_loss = pref_loss / pref_bs
                    pref_loss.backward()

                    if (batch_id + 1) % accumulation_steps == 0:
                        hypernet_optimizer.step()
                        hypernet_optimizer.zero_grad()
                        model.zero_grad()

                    gpu_end_time = time.time()
                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    epoch_total_loss += pref_loss.item()

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs

                    gc.collect()
                    batch_id += 1

                average_one_sample_loss_in_epoch = epoch_total_loss / max(epoch_total_size, 1)
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            with torch.no_grad():
                alpha_test = torch.tensor([tau_p], device=device)
                clf_weight, clf_bias = local_hypernetwork(alpha_test)
                apply_clf_weights(model, clf_weight, clf_bias, param_dict["task"])

            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            torch.save(model.cpu(), client_model_path)

            client_hypernet_path = os.path.join(basic_path, "client_" + str(id + 1), 'hypernetwork.pt')
            torch.save(local_hypernetwork.cpu(), client_hypernet_path)

            del model, local_hypernetwork
            gc.collect()

        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        logger.info("Parameter aggregation - Base Model")
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

        logger.info("Parameter aggregation - HyperNetwork")
        hypernet_theta_list = []
        for id in idxs_users:
            client_hypernet_path = os.path.join(basic_path, "client_" + str(id + 1), 'hypernetwork.pt')
            client_hypernet = torch.load(client_hypernet_path, weights_only=False)
            hypernet_theta_list.append(get_parameters(client_hypernet))
            del client_hypernet
            gc.collect()

        hypernet_theta_list = np.array(hypernet_theta_list, dtype=object)
        hypernet_theta_avg = np.average(hypernet_theta_list, axis=0,
                                        weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        set_parameters(hypernetwork, hypernet_theta_avg)

        with torch.no_grad():
            alpha_final = torch.tensor([tau_p], device=device)
            hypernetwork.to(device)
            clf_weight, clf_bias = hypernetwork(alpha_final)
            hypernetwork.cpu()
            apply_clf_weights(global_model, clf_weight.cpu(), clf_bias.cpu(), param_dict["task"])

        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list, hypernet_theta_list
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
                    'hypernetwork_state': hypernetwork.state_dict()
                }
            )
            clean_old_checkpoints(param_dict, keep_latest=5)

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_PraFFL.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
