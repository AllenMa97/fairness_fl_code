import copy
import os
import gc
import time
import torch
import numpy as np
import random
import itertools
import torch.nn.functional as F
from scipy.special.cython_special import logit
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from tool.logger import *
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, get_HM_by_two_value




class CustomDataset(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """

    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""

        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor

    def __getitem__(self, index):
        """Returns the selected data based on the index information."""

        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""

        return len(self.x)


class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).

    This class is for implementing the lambda adjustment and batch selection of FairBatch.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values in FairBatch.


    """

    def __init__(self, param_dict, model, x_tensor, y_tensor, z_tensor, batch_size, alpha, target_fairness, replacement=False,
                 seed=0):
        """Initializes FairBatch."""
        self.param_dict = param_dict
        self.model = model

        # np.random.seed(seed)
        # random.seed(seed)

        self.x_data = x_tensor
        self.y_data = y_tensor
        self.z_data = z_tensor

        self.alpha = alpha
        self.fairness_type = target_fairness
        self.replacement = replacement

        self.N = len(z_tensor)

        self.batch_size = batch_size
        self.batch_num = int(len(self.y_data) / self.batch_size)

        # Takes the unique values of the tensors
        self.z_item = list(set(z_tensor.tolist()))
        self.y_item = list(set(y_tensor.tolist()))

        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))

        # Makes masks
        self.z_mask = {}
        self.y_mask = {}
        self.yz_mask = {}

        for tmp_z in self.z_item:
            self.z_mask[tmp_z] = (self.z_data == tmp_z)

        for tmp_y in self.y_item:
            self.y_mask[tmp_y] = (self.y_data == tmp_y)

        for tmp_yz in self.yz_tuple:
            self.yz_mask[tmp_yz] = (self.y_data == tmp_yz[0]) & (self.z_data == tmp_yz[1])

        # Finds the index
        self.z_index = {}
        self.y_index = {}
        self.yz_index = {}

        for tmp_z in self.z_item:
            self.z_index[tmp_z] = (self.z_mask[tmp_z] == 1).nonzero().squeeze()

        for tmp_y in self.y_item:
            self.y_index[tmp_y] = (self.y_mask[tmp_y] == 1).nonzero().squeeze()

        for tmp_yz in self.yz_tuple:
            self.yz_index[tmp_yz] = (self.yz_mask[tmp_yz] == 1).nonzero().squeeze()

        # Length information
        self.z_len = {}
        self.y_len = {}
        self.yz_len = {}

        for tmp_z in self.z_item:
            try:  # BUG
                self.z_len[tmp_z] = len(self.z_index[tmp_z])
            except TypeError:
                self.z_len[tmp_z] = 1

        for tmp_y in self.y_item:
            try:  # BUG
                self.y_len[tmp_y] = len(self.y_index[tmp_y])
            except TypeError:
                self.y_len[tmp_y] = 0

        for tmp_yz in self.yz_tuple:  # BUG
            if len(self.yz_index[tmp_yz].size()) == 0:
                self.yz_len[tmp_yz] = 1
            else:
                self.yz_len[tmp_yz] = len(self.yz_index[tmp_yz])

        # Default batch size
        # self.S = {}  # BUG
        self.S = {(0, 0): 0, (1, 0): 0, (0, 1): 0, (1, 1): 0}

        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * (self.yz_len[tmp_yz]) / self.N

        # BUG
        try:
            self.lb1 = (self.S[1, 1]) / (self.S[1, 1] + (self.S[1, 0]))
        except Exception:
            self.lb1 = 0
        try:
            # self.lb2 = (self.S[-1, 1]) / (self.S[-1, 1] + (self.S[-1, 0]))  # BUG
            self.lb2 = (self.S[0, 1]) / (self.S[0, 1] + (self.S[0, 0]))
        except Exception:
            self.lb2 = 0

    def adjust_lambda(self):
        """Adjusts the lambda values for FairBatch algorithm.

        The detailed algorithms are decribed in the paper.

        """
        param_dict = self.param_dict
        self.model.eval()
        # logit = self.model(self.x_data)

        # 为了BERTCLASSIFIER专门修改
        with torch.no_grad():
            # 这个地方也从一开始的全部数据同时读取改为一次性只读取一部分，避免爆显存
            logit_list = []
            for i in range(0, self.batch_num+1):
                if "SENT_CLF" in param_dict["task"]:
                    _, tmp_logit = self.model(
                        input_ids=self.x_data[i*self.batch_size: (i+1)*self.batch_size, 0],
                        attention_mask=self.x_data[i*self.batch_size:(i+1)*self.batch_size, 1]
                    )
                    logit_list.append(tmp_logit)
                    del _, tmp_logit
                torch.cuda.empty_cache()
            logit = torch.concatenate(logit_list)
            if "SENT_CLF" in param_dict["task"]:
                criterion = torch.nn.CrossEntropyLoss(reduction='none')

            if self.fairness_type == 'eqopp':

                yhat_yz = {}
                yhat_y = {}

                # eo_loss = criterion((F.tanh(logit) + 1) / 2, (self.y_data + 1) / 2)
                # eo_loss = criterion((F.tanh(logit) + 1) / 2, (self.y_data.reshape(-1, 1).float() + 1) / 2)
                eo_loss = criterion((F.tanh(logit) + 1) / 2, ((self.y_data + 1) / 2).long())

                for tmp_yz in self.yz_tuple:
                    if self.yz_len[tmp_yz] != 0:  # BUG
                        yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]
                    else:
                        yhat_yz[tmp_yz] = 0

                for tmp_y in self.y_item:
                    if self.y_len[tmp_y] != 0:  # BUG
                        yhat_y[tmp_y] = float(torch.sum(eo_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]
                    else:
                        yhat_y[tmp_y] = 0

                # lb1 * loss_z1 + (1-lb1) * loss_z0

                try:  # BUG
                    if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                        self.lb1 += self.alpha
                    else:
                        self.lb1 -= self.alpha
                except KeyError:
                    if (1, 1) not in yhat_yz:
                        yhat_yz[(1, 1)] = 0
                    if (1, 0) not in yhat_yz:
                        yhat_yz[(1, 0)] = 0
                    if (0, 1) not in yhat_yz:
                        yhat_yz[(0, 1)] = 0
                    if (0, 0) not in yhat_yz:
                        yhat_yz[(0, 0)] = 0

                    if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                        self.lb1 += self.alpha
                    else:
                        self.lb1 -= self.alpha

                if self.lb1 < 0:
                    self.lb1 = 0
                elif self.lb1 > 1:
                    self.lb1 = 1

            elif self.fairness_type == 'eqodds':

                yhat_yz = {}
                yhat_y = {}

                # eo_loss = criterion((F.tanh(logit) + 1) / 2, (self.y_data + 1) / 2)
                eo_loss = criterion((F.tanh(logit) + 1) / 2, (self.y_data.reshape(-1, 1).float() + 1) / 2)

                for tmp_yz in self.yz_tuple:
                    if self.yz_len[tmp_yz] == 0:  # BUG
                        yhat_yz[tmp_yz] = float(torch.sum(eo_loss[self.yz_index[tmp_yz]])) / self.yz_len[tmp_yz]
                    else:
                        yhat_yz[tmp_yz] = 0

                for tmp_y in self.y_item:
                    yhat_y[tmp_y] = float(torch.sum(eo_loss[self.y_index[tmp_y]])) / self.y_len[tmp_y]

                y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
                # y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])  # BUG
                y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])

                # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                if y1_diff > y0_diff:
                    if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                        self.lb1 += self.alpha
                    else:
                        self.lb1 -= self.alpha
                else:
                    # if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:  # BUG
                    if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
                        self.lb2 += self.alpha
                    else:
                        self.lb2 -= self.alpha

                if self.lb1 < 0:
                    self.lb1 = 0
                elif self.lb1 > 1:
                    self.lb1 = 1

                if self.lb2 < 0:
                    self.lb2 = 0
                elif self.lb2 > 1:
                    self.lb2 = 1

            elif self.fairness_type == 'dp':
                yhat_yz = {}
                yhat_y = {}

                ones_array = np.ones(len(self.y_data))
                ones_tensor = torch.FloatTensor(ones_array)
                dp_loss = criterion((F.tanh(logit) + 1) / 2, ones_tensor)  # Note that ones tensor puts as the true label

                for tmp_yz in self.yz_tuple:
                    yhat_yz[tmp_yz] = float(torch.sum(dp_loss[self.yz_index[tmp_yz]])) / self.z_len[tmp_yz[1]]

                y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
                # y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])  # BUG
                y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])

                # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                if y1_diff > y0_diff:
                    if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                        self.lb1 += self.alpha
                    else:
                        self.lb1 -= self.alpha
                else:
                    # if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:  # BUG
                    if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
                        self.lb2 -= self.alpha
                    else:
                        self.lb2 += self.alpha

                if self.lb1 < 0:
                    self.lb1 = 0
                elif self.lb1 > 1:
                    self.lb1 = 1

                if self.lb2 < 0:
                    self.lb2 = 0
                elif self.lb2 > 1:
                    self.lb2 = 1

            torch.cuda.empty_cache()

    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement=False):
        """Selects a certain number of batches based on the given batch size.

        Args:
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.

        Returns:
            Indices that indicate the data.

        """

        select_index = []

        if replacement == True:
            for _ in range(batch_num):
                select_index.append(np.random.choice(full_index, batch_size, replace=False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            try:  # BUG
                random.shuffle(tmp_index)
            except:
                tmp_index = np.array([tmp_index])

            start_idx = 0
            for i in range(batch_num):
                try:
                    len_of_full_index = len(full_index)
                except Exception:
                    len_of_full_index = 1
                if start_idx + batch_size > len_of_full_index:
                    select_index.append(np.concatenate(
                        (tmp_index[start_idx:], tmp_index[: batch_size - (len_of_full_index - start_idx)])))

                    start_idx = len_of_full_index - start_idx
                else:

                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size

        return select_index

    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.

        Returns:
            Indices that indicate the data in each batch.

        """

        if self.fairness_type == 'original':

            entire_index = torch.FloatTensor([i for i in range(len(self.y_data))])

            sort_index = self.select_batch_replacement(self.batch_size, entire_index, self.batch_num, self.replacement)

            for i in range(self.batch_num):
                yield sort_index[i]

        else:

            self.adjust_lambda()  # Adjust the lambda values
            each_size = {}

            # Based on the updated lambdas, determine the size of each class in a batch
            if self.fairness_type == 'eqopp':
                # lb1 * loss_z1 + (1-lb1) * loss_z0

                each_size[(1, 1)] = round(self.lb1 * (self.S[(1, 1)] + self.S[(1, 0)]))
                each_size[(1, 0)] = round((1 - self.lb1) * (self.S[(1, 1)] + self.S[(1, 0)]))
                # each_size[(-1, 1)] = round(self.S[(-1, 1)])
                each_size[(0, 1)] = round(self.S[(0, 1)])
                # each_size[(-1, 0)] = round(self.S[(-1, 0)])
                each_size[(0, 0)] = round(self.S[(0, 0)])

            elif self.fairness_type == 'eqodds':
                # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                each_size[(1, 1)] = round(self.lb1 * (self.S[(1, 1)] + self.S[(1, 0)]))
                each_size[(1, 0)] = round((1 - self.lb1) * (self.S[(1, 1)] + self.S[(1, 0)]))
                # each_size[(-1, 1)] = round(self.lb2 * (self.S[(-1, 1)] + self.S[(-1, 0)]))
                each_size[(0, 1)] = round(self.lb2 * (self.S[(0, 1)] + self.S[(0, 0)]))
                # each_size[(-1, 0)] = round((1 - self.lb2) * (self.S[(-1, 1)] + self.S[(-1, 0)]))
                each_size[(0, 0)] = round((1 - self.lb2) * (self.S[(0, 1)] + self.S[(0, 0)]))

            elif self.fairness_type == 'dp':
                # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
                # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

                each_size[(1, 1)] = round(self.lb1 * (self.S[(1, 1)] + self.S[(1, 0)]))
                each_size[(1, 0)] = round((1 - self.lb1) * (self.S[(1, 1)] + self.S[(1, 0)]))
                # each_size[(-1, 1)] = round(self.lb2 * (self.S[(-1, 1)] + self.S[(-1, 0)]))
                each_size[(0, 1)] = round(self.lb2 * (self.S[(0, 1)] + self.S[(0, 0)]))
                # each_size[(-1, 0)] = round((1 - self.lb2) * (self.S[(-1, 1)] + self.S[(-1, 0)]))
                each_size[(0, 0)] = round((1 - self.lb2) * (self.S[(0, 1)] + self.S[(0, 0)]))

            # Get the indices for each class
            try:  # BUG
                sort_index_y_1_z_1 = self.select_batch_replacement(each_size[(1, 1)], self.yz_index[(1, 1)],
                                                                   self.batch_num,
                                                                   self.replacement)
            except KeyError:
                sort_index_y_1_z_1 = [np.array([]) for _ in range(self.batch_num)]

            try:  # BUG
                # sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(-1, 1)], self.yz_index[(-1, 1)],
                #                                                    self.batch_num, self.replacement)
                sort_index_y_0_z_1 = self.select_batch_replacement(each_size[(0, 1)], self.yz_index[(0, 1)],
                                                                   self.batch_num, self.replacement)
            except KeyError:
                sort_index_y_0_z_1 = [np.array([]) for _ in range(self.batch_num)]

            try:  # BUG
                sort_index_y_1_z_0 = self.select_batch_replacement(each_size[(1, 0)], self.yz_index[(1, 0)],
                                                                   self.batch_num,
                                                                   self.replacement)
            except KeyError:
                sort_index_y_1_z_0 = [np.array([]) for _ in range(self.batch_num)]

            try:
                # sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(-1, 0)], self.yz_index[(-1, 0)],
                #                                                    self.batch_num, self.replacement)
                sort_index_y_0_z_0 = self.select_batch_replacement(each_size[(0, 0)], self.yz_index[(0, 0)],
                                                                   self.batch_num, self.replacement)
            except KeyError:
                sort_index_y_0_z_0 = [np.array([]) for _ in range(self.batch_num)]

            for i in range(self.batch_num):
                try:
                    key_in_fairbatch = sort_index_y_0_z_0[i].copy()
                    key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
                    key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
                    key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))
                except:
                    print(3)

                random.shuffle(key_in_fairbatch)

                # 自行添加的代码，防止算法采样数超过实验设置的batch_size导致爆显存
                key_in_fairbatch = key_in_fairbatch[:self.batch_size]

                yield key_in_fairbatch

    def __len__(self):
        """Returns the length of data."""

        return len(self.y_data)


def construct_fairbatch_dataset(device, client_training_dataset):
    try:
        indices = client_training_dataset.indices.tolist()
    except Exception:
        indices = client_training_dataset.indices

    x0_list, x1_list, y_list, z_list = [], [], [], []
    for item in indices:
        x0_list.append(client_training_dataset.dataset[item]['input_ids'])
        x1_list.append(client_training_dataset.dataset[item]['attention_mask'])

        y_list.append(client_training_dataset.dataset[item]['labels'])
        z_list.append(client_training_dataset.dataset[item]['protected'])

    x0 = torch.stack(x0_list)
    x1 = torch.stack(x1_list)
    x = torch.stack([x0, x1], dim=1)
    y = torch.stack(y_list)
    z = torch.stack(z_list)

    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    fairbatch_dataset = CustomDataset(x, y, z)
    return fairbatch_dataset

def construct_the_fairsampler(param_dict, model, client_training_dataset, batch_size, alpha, target_fairness):
    if ("oportunity" in target_fairness) or ("eqopp" in target_fairness):
        # case 1: Equal opportunity
        target_fairness = 'eqopp'
    elif ("odds" in target_fairness) or ("eqodds" in target_fairness):
        # case 2: Equalized odds
        target_fairness = 'eqodds'
    else:
        # case 3: Demographic parity
        target_fairness = 'dp'

    sampler = FairBatch(param_dict, model, client_training_dataset.x, client_training_dataset.y, client_training_dataset.z, batch_size,
                        alpha, target_fairness=target_fairness, replacement=False)
    return sampler



def FL_FairBatch(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(num_clients_K) + "Clients")

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")


    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )


        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        # Simulate Client Parallel
        for id in idxs_users:
            # Local Initialization
            # 下发模型
            logger.info("Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))

            client_i_dataset = client_dataset_list[id]
            client_i_fairbatch_dataset = construct_fairbatch_dataset(device, client_i_dataset)
            fair_sampler = construct_the_fairsampler(param_dict, model, client_i_fairbatch_dataset,
                                                     param_dict['batch_size'], 0.005, 'eqopp')

            client_i_dataloader = torch.utils.data.DataLoader(client_i_fairbatch_dataset, sampler=fair_sampler, num_workers=0)

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                # for batch_index, batch in enumerate(client_i_dataloader):

                # 记录GPU计算开始时间
                gpu_start_time = time.time()

                for batch in client_i_dataloader:
                    # input_ids尺寸 [batch_size, max_len]
                    # input_ids = batch["input_ids"].to(device)
                    input_ids = batch[0][0,:,0].to(device)
                    # attention_mask = batch["attention_mask"].to(device)
                    attention_mask = batch[0][0,:,1].to(device)

                    # labels尺寸 [batch_size]
                    # labels = batch["labels"].to(device)
                    labels = batch[1][0].to(device)

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size


                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)
                    loss = torch.sum(batch_loss)
                    if loss.item() != 0:
                        loss = loss / true_batch_size
                        loss.backward()

                    # FedAvg算法一个batch就做一次更新
                    optimizer.step()

                    # 清空梯度
                    model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    del input_ids, attention_mask, labels
                    gc.collect()

                # 记录GPU计算结束时间
                gpu_end_time = time.time()

                users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                epoch_total_size = max(epoch_total_size, client_datasets_size_list[id])

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            del model, fair_sampler
            gc.collect()
            torch.cuda.empty_cache()

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # Global operation
        logger.info("Parameter aggregation")
        theta_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            theta_list.append(get_parameters(selected_model))
            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        # FedAvg新版论文的聚合权重是数据占比
        # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
        # 如一个weights = [w1, w2, w3, w4]
        # 那么结果就是(theta1 * w1 + theta2 * w2 + theta3 * w3 + theta4 * w4)/ sum(w1+w2+w3+w4)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        # FedAvg旧版论文的聚合权重是平均
        # theta_avg = np.mean(theta_list, 0).tolist()
        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)

        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I']:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
            logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FL_FairBatch.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
