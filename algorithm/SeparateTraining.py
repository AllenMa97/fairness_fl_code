import os
import torch
import copy
import math
import numpy as np
import gc
from tool.logger import *
from tool.utils import get_parameters, set_parameters, save_model, save_model_sepa
from transformers import AdamW, get_linear_schedule_with_warmup
from algorithm.Optimizers import BERTCLF_Optimizer


def ST_BertClassifier(device,
                      global_model,
                      algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
                      training_dataloaders,
                      training_dataset,
                      client_dataset_list,
                      param_dict,
                      testing_dataloader=None):

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    del training_dataset, client_dataset_list
    gc.collect()

    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(num_clients_K) + "Clients")

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]): # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Simulate Client Parallel
    EPOCHS = algorithm_epoch_T * communication_round_I

    for id in range(num_clients_K):
        client_i_dataloader = training_dataloaders[id]
        client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
        model = torch.load(client_model_path, weights_only=False) # 持久化
        # model = local_model_list[id] # 内存化
        model.train()
        model.to(device)

        optimizer = BERTCLF_Optimizer(
            method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
        optimizer.set_parameters(list(model.named_parameters()))
        total_steps = len(client_i_dataloader) * EPOCHS

        # Local Training
        for epoch in range(EPOCHS):
            logger.info(f'------ Epoch {epoch + 1}/{EPOCHS} ------')

            losses = []
            correct_predictions = 0

            for index, d in enumerate(client_i_dataloader):
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)

                features, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                activated_preds = logits.softmax(dim=1)
                _, preds = torch.max(activated_preds, dim=1)
                loss = criterion(activated_preds, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_acc, train_loss = correct_predictions.double() / training_dataset_size, np.mean(losses)


            logger.info(f"##########SeparateTraining"
                        f"Client: {id+1} / {num_clients_K}; "
                        f"Epoch: {epoch + 1}; "
                        f"Train loss {train_loss}; "
                        f"Accuracy {train_acc}; "
                        f"####")

        # Upgrade the local model list

        torch.save(model.cpu(), client_model_path) # 持久化
        # local_model_list[id] = model.cpu() # 内存化
        del model
        torch.cuda.empty_cache()

    logger.info("Training finish, return global model and local model list")
    # return local_model_list
