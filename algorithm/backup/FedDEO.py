# FedDEO: Description-Enhanced One-Shot Federated Learning with Diffusion Models
# https://arxiv.org/pdf/2407.19953 (ACM MM 2024)
# 核心机制依赖扩散模型生成图像数据，仅适用于图像分类任务
# 做Latent SD任务，在各个本地训练多个的文本输入向量x并上传到服务器，服务器根据各个文本输入向量x生成图像数据，再用图像数据训练全局模型

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
from hypothesis.generator import FigGenerator


class DescriptionVector(nn.Module):
    def __init__(self, description_dim, num_classes=2):
        super(DescriptionVector, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(num_classes, description_dim))
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self, labels):
        return self.embeddings[labels.long()]


def FedDEO(device,
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
            f"FedDEO relies on diffusion models to generate image data from descriptions, "
            f"which is only applicable to IMG_CLF. "
            f"Current task: {param_dict['task']}. "
            f"FedDEO does not support SENT_CLF or Tabular_CLF."
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

    description_dim = 100

    client_descriptions = []
    client_label_distributions = []

    for id in idxs_users:
        logger.info(f"Client {id}: Training local model and description vectors")
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

        logger.info(f"Client {id}: Training description vectors")
        desc_vector = DescriptionVector(description_dim, num_classes=2).to(device)
        desc_optimizer = torch.optim.Adam(desc_vector.parameters(), lr=param_dict['learning_rate'])

        model = model.to(device)
        model.eval()

        label_count = [0, 0]
        with torch.no_grad():
            for batch in client_i_dataloader:
                labels = batch["labels"]
                for l in labels:
                    label_count[int(l)] += 1

        total_labels = sum(label_count)
        client_label_distributions.append([c / total_labels for c in label_count])

        desc_epochs = 30
        for desc_epoch in range(desc_epochs):
            total_desc_loss = 0
            num_batches = 0

            for batch in client_i_dataloader:
                labels = batch["labels"].to(device)
                descriptions = desc_vector(labels)

                with torch.no_grad():
                    imgs = batch["img"].to(device)
                    client_logits, _ = model(imgs)
                    client_probs = torch.cat([1 - client_logits, client_logits], dim=1)

                desc_loss = F.mse_loss(descriptions, client_probs.detach())
                desc_optimizer.zero_grad()
                desc_loss.backward()
                desc_optimizer.step()

                total_desc_loss += desc_loss.item()
                num_batches += 1

            if (desc_epoch + 1) % 10 == 0:
                logger.info(f"Client {id} Desc Epoch {desc_epoch + 1}/{desc_epochs}, Avg Loss: {total_desc_loss / max(num_batches, 1):.4f}")

        client_descriptions.append(desc_vector.cpu())
        del model, desc_vector, desc_optimizer
        gc.collect()
        torch.cuda.empty_cache()

    total_gpu_seconds += sum(users_gpu_seconds_list)
    logger.info(f"Phase 1 Communication Cost: {num_clients_K * 2 * model_MB_size} MB")

    logger.info("FedDEO Server: Generate Synthetic Data from Descriptions and Train Global Model")
    gpu_start_time = time.time()

    Generator = FigGenerator(nz=description_dim, ngf=64, img_size=64, nc=3).to(device)

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
            desc_vector = client_descriptions[cid_idx].to(device)
            label_dist = client_label_distributions[cid_idx]

            num_samples = batch_size
            sampled_labels = torch.multinomial(torch.tensor(label_dist), num_samples, replacement=True).to(device)

            with torch.no_grad():
                descriptions = desc_vector(sampled_labels)
                noise = torch.randn_like(descriptions)
                synthetic_data = Generator(noise)

            global_pred, _ = global_model(synthetic_data)
            criterion = torch.nn.BCELoss()
            server_loss = criterion(global_pred[:, 0], sampled_labels.float())

            server_loss.backward()
            global_optimizer.step()
            global_model.zero_grad()
            global_optimizer.zero_grad()

            del desc_vector, descriptions, noise, synthetic_data, sampled_labels, server_loss
            gc.collect()
            torch.cuda.empty_cache()

    gpu_end_time = time.time()
    total_gpu_seconds += (gpu_end_time - gpu_start_time)

    avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
    logger.info(f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

    logger.info("Training finish, save and return the global model.")
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedDEO.pt")
    torch.save(global_model, save_path)

    desc_MB_size = sum(p.numel() for p in client_descriptions[0].parameters()) * 4 / (1024 * 1024)
    total_communication_cost = num_clients_K * model_MB_size * 2 + num_clients_K * desc_MB_size * 2
    return global_model, total_gpu_seconds, total_communication_cost
