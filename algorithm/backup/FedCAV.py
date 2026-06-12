# DATA-FREE ONE-SHOT FEDERATED LEARNING UNDER VERY HIGH STATISTICAL HETEROGENEITY
# https://openreview.net/forum?id=_hb4vM3jspB (ICLR 2023)
# 需要模型具有分离的特征提取器和分类头（only_backbone_forward + only_clf_forward）
# 需要练一个CVAE，十分不稳定导致昂贵；需要上传本地标签分布信息，容易侵犯隐私；本地不进行

import copy
import os
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer


class CVAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, label_dim=2):
        super(CVAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + label_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, y_onehot):
        x = torch.cat([x, y_onehot], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, label_dim=2):
        super(CVAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + label_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, z, y_onehot):
        z = torch.cat([z, y_onehot], dim=-1)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def FedCAV(device,
           global_model,
           algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
           training_dataloaders,
           training_dataset,
           client_dataset_list,
           param_dict,
           testing_dataloader,
           testing_dataset_len
           ):

    if "Tabular_CLF" in param_dict["task"] and "LogisticRegression" in param_dict.get("model_type", ""):
        raise ValueError(
            f"FedCAV requires a model with separate feature extractor and classifier head "
            f"(only_backbone_forward + only_clf_forward). "
            f"LogisticRegression does not have a feature extractor. "
            f"Please use ANN model type for Tabular_CLF, or use SENT_CLF / IMG_CLF."
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

    if "SENT_CLF" in param_dict["task"]:
        feature_dim = 768
    elif "IMG_CLF" in param_dict["task"]:
        feature_dim = 512
    elif "Tabular_CLF" in param_dict["task"]:
        feature_dim = param_dict.get('nn_input_size', 128)

    latent_dim = 64
    label_dim = 2

    client_decoders = []
    client_label_distributions = []

    for id in idxs_users:
        logger.info(f"Client {id}: Training CVAE on local data")
        model = copy.deepcopy(global_model)
        model.eval()
        model.to(device)
        client_i_dataloader = training_dataloaders[id]

        encoder = CVAEEncoder(feature_dim, latent_dim, label_dim).to(device)
        decoder = CVAEDecoder(latent_dim, feature_dim, label_dim).to(device)
        cvae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                          lr=param_dict['learning_rate'])

        label_count = [0, 0]
        features_list = []
        labels_list = []

        with torch.no_grad():
            for batch in client_i_dataloader:
                if "SENT_CLF" in param_dict["task"]:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    features = model.only_PLM_forward(input_ids=input_ids, attention_mask=attention_mask)
                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                    features = model.only_backbone_forward(imgs)
                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    features = model.only_backbone_forward(X)

                labels = batch["labels"]
                features_list.append(features.cpu())
                labels_list.append(labels.cpu())
                for l in labels:
                    label_count[int(l)] += 1

        all_features = torch.cat(features_list, dim=0).to(device)
        all_labels = torch.cat(labels_list, dim=0).to(device)
        total_labels = sum(label_count)
        label_distribution = [c / total_labels for c in label_count]
        client_label_distributions.append(label_distribution)

        cvae_epochs = 50
        batch_size = param_dict['batch_size']
        dataset_size = len(all_labels)
        recon_criterion = torch.nn.MSELoss()

        for cvae_epoch in range(cvae_epochs):
            indices = torch.randperm(dataset_size)
            total_cvae_loss = 0
            num_batches = 0

            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_features = all_features[batch_indices]
                batch_labels = all_labels[batch_indices]
                y_onehot = F.one_hot(batch_labels.long(), label_dim).float()

                mu, logvar = encoder(batch_features, y_onehot)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                reconstructed = decoder(z, y_onehot)
                recon_loss = recon_criterion(reconstructed, batch_features)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                cvae_loss = recon_loss + 0.1 * kl_loss

                cvae_optimizer.zero_grad()
                cvae_loss.backward()
                cvae_optimizer.step()

                total_cvae_loss += cvae_loss.item()
                num_batches += 1

            if (cvae_epoch + 1) % 10 == 0:
                logger.info(f"Client {id} CVAE Epoch {cvae_epoch + 1}/{cvae_epochs}, Avg Loss: {total_cvae_loss / max(num_batches, 1):.4f}")

        client_decoders.append(decoder.cpu())
        del model, encoder, decoder, cvae_optimizer, all_features, all_labels, features_list, labels_list
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("FedCAV Server: Ensemble Decoders for Global Model Training")
    gpu_start_time = time.time()

    global_model = global_model.to(device)
    global_optimizer = BERTCLF_Optimizer(method=param_dict['optimize_method'],
                                         learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    global_optimizer.set_parameters(list(global_model.named_parameters()))

    server_training_steps = 200
    batch_size = param_dict['batch_size']

    for s_step in range(server_training_steps):
        if s_step % 50 == 0:
            logger.info(f"Server training step: {s_step + 1}/{server_training_steps}")

        for cid_idx, cid in enumerate(idxs_users):
            decoder = client_decoders[cid_idx].to(device)
            label_dist = client_label_distributions[cid_idx]

            num_samples = batch_size
            sampled_labels = torch.multinomial(torch.tensor(label_dist), num_samples, replacement=True).to(device)
            y_onehot = F.one_hot(sampled_labels.long(), label_dim).float()
            z = torch.randn(num_samples, latent_dim, device=device)

            with torch.no_grad():
                synthetic_features = decoder(z, y_onehot)

            _, global_output = global_model.only_clf_forward(synthetic_features)

            if "SENT_CLF" in param_dict["task"]:
                criterion = torch.nn.CrossEntropyLoss()
                server_loss = criterion(global_output, sampled_labels.long())
            elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                criterion = torch.nn.BCELoss()
                server_loss = criterion(global_output[:, 0], sampled_labels.float())

            server_loss.backward()
            global_optimizer.step()
            global_model.zero_grad()
            global_optimizer.zero_grad()

            del decoder, synthetic_features, z, y_onehot, sampled_labels, server_loss
            gc.collect()
            torch.cuda.empty_cache()

    gpu_end_time = time.time()
    total_gpu_seconds += (gpu_end_time - gpu_start_time)

    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedCAV.pt")
    torch.save(global_model, save_path)

    total_communication_cost = num_clients_K * model_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost
