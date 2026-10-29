# FAIRBATCH:BATCH SELECTION FOR MODEL FAIRNESS
# https://openreview.net/pdf?id=YNnpaAKeCfx
# 核心思想：将FairBatch技术应用于联邦学习场景，通过公平的批处理策略来实现组间公平性

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
from tool.aggregation import average_state_dicts
from tool.utils import get_parameters, set_parameters
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import save_checkpoint, clean_old_checkpoints
from tool.amp_utils import autocast_context, get_scaler, scale_backward, scaler_step
from tool.client_parallel import ClientParallelExecutor
from tool.tensorboard_logger import log_scalar, log_metrics, log_test_metrics, log_system_metrics, update_step, flush, log_deep_metrics, get_monitoring_config


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

        # x_data 常驻 CPU（见 construct_fairbatch_dataset）：整份客户端数据搬到 GPU
        # 会在 FairFace 等大数据集上 OOM，这里改为逐 batch 上卡推理、结果回 CPU，
        # 使 GPU 上瞬时只保留一个 batch 的特征。
        device = next(self.model.parameters()).device

        # 为了BERTCLASSIFIER专门修改
        with torch.no_grad():
            logit_list = []
            for i in range(0, self.batch_num+1):
                if "SENT_CLF" in param_dict["task"]:
                    _, tmp_logit = self.model(
                        input_ids=self.x_data[i*self.batch_size: (i+1)*self.batch_size, 0].to(device),
                        attention_mask=self.x_data[i*self.batch_size:(i+1)*self.batch_size, 1].to(device)
                    )
                    logit_list.append(tmp_logit.cpu())
                    del _, tmp_logit
                elif "Tabular_CLF" in param_dict["task"]:
                    tmp_logit, _ = self.model(self.x_data[i*self.batch_size: (i+1)*self.batch_size].to(device))
                    logit_list.append(tmp_logit.cpu())
                    del tmp_logit
                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    tmp_logit, _ = self.model(self.x_data[i*self.batch_size: (i+1)*self.batch_size].to(device))
                    logit_list.append(tmp_logit.cpu())
                    del tmp_logit
                torch.cuda.empty_cache()
            logit = torch.concatenate(logit_list)
            if "SENT_CLF" in param_dict["task"]:
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
            elif "Tabular_CLF" in param_dict["task"]:
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

            if self.fairness_type == 'eqopp':

                yhat_yz = {}
                yhat_y = {}

                if "SENT_CLF" in param_dict["task"]:
                    eo_loss = criterion((F.tanh(logit) + 1) / 2, ((self.y_data + 1) / 2).long())
                elif "Tabular_CLF" in param_dict["task"]:
                    # 用 reshape(-1) 而非 squeeze()：squeeze() 在 batch=1 时会把
                    # [1] 压成 0 维标量，导致 BCEWithLogitsLoss 报
                    # "Target size ([1]) must be the same as input size ([])"。
                    eo_loss = criterion(logit[:, 0].reshape(-1), self.y_data.float().reshape(-1))
                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    eo_loss = criterion(logit[:, 0].reshape(-1), self.y_data.float().reshape(-1))

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

        # adjust_lambda 用 eval 模式做推断是对的（BN 不更新统计量、dropout 关闭），
        # 但它在 __iter__ 中被调用，若不恢复训练模式，整个 DataLoader 迭代期间模型会
        # 一直停在 eval 状态，使后续训练用错 BN 统计量。这里显式恢复 train 模式。
        self.model.train()

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


def construct_fairbatch_dataset(device, client_training_dataset, param_dict):
    try:
        indices = client_training_dataset.indices.tolist()
    except Exception:
        indices = client_training_dataset.indices

    y_list, z_list = [], []
    if "SENT_CLF" in param_dict["task"]:
        x0_list, x1_list = [], []
        for item in indices:
            x0_list.append(client_training_dataset.dataset[item]['input_ids'])
            x1_list.append(client_training_dataset.dataset[item]['attention_mask'])
            y_list.append(torch.tensor(client_training_dataset.dataset[item]['labels']))
            z_list.append(torch.tensor(client_training_dataset.dataset[item]['protected']))
        x0 = torch.stack(x0_list)
        x1 = torch.stack(x1_list)
        x = torch.stack([x0, x1], dim=1)
    elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
        x_list = []
        for item in indices:
            x_list.append(client_training_dataset.dataset[item]['img'])
            y_list.append(torch.tensor(client_training_dataset.dataset[item]['labels']))
            z_list.append(torch.tensor(client_training_dataset.dataset[item]['protected']))
        x = torch.stack(x_list)
    elif "Tabular_CLF" in param_dict["task"]:
        x_list = []
        for item in indices:
            x_list.append(client_training_dataset.dataset[item]['X'])
            y_list.append(torch.tensor(client_training_dataset.dataset[item]['labels']))
            z_list.append(torch.tensor(client_training_dataset.dataset[item]['protected']))
        x = torch.stack(x_list).unsqueeze(1)

    y = torch.stack(y_list)
    z = torch.stack(z_list)

    # 保持 CPU 常驻：整份客户端数据直接 .to(device) 会在 FairFace 等大数据集上
    # OOM（单客户端 x 可达 ~11GiB）。训练时 DataLoader 逐 batch .to(device)，
    # FairBatch.adjust_lambda 亦逐 batch 上卡推理，GPU 上只需容纳一个 batch。
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


def _train_single_client_fairbatch(client_id, device, model, param_dict, training_dataloaders,
                                   algorithm_epoch_T, client_datasets_size_list, num_clients_K,
                                   basic_path, iter_t, communication_round_I,
                                   client_dataset_list, global_lb1, global_lb2,
                                   use_amp, scaler, criterion):
    """FL_FairBatch 单客户端训练函数（供 ClientParallelExecutor 调用）"""
    logger.info("Copy From Global Model")
    model.train()
    model.to(device)
    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(model.named_parameters()))

    client_i_dataset = client_dataset_list[client_id]
    client_i_fairbatch_dataset = construct_fairbatch_dataset(device, client_i_dataset, param_dict)
    # 显存优化：把单批样本数从 256 降到 64，并用梯度累积保持「每 256 个样本一次参数
    # 更新」的原语义。峰值激活与单批大小成正比，降到 1/4 后峰值同步降到约 1/4。
    # 梯度累积：每个 micro-batch 反传但不 step，累计 accumulation_steps 次后再统一
    # step 与清零，等价于在合并后的大批上做一次更新（损失按样本数取平均时严格等价）。
    #
    # ===== 显存踩坑实测记录（勿删，供后续排障对照）=====
    # 1) 历史峰值（batch=256/FP32，图像数据集）：
    #    - UTKFace / FairFace 单进程实测 37~41 GiB，会挤爆同卡其它作业；
    #      其中一个持有者是 `epoch_total_loss += loss` 带图累加（整 epoch 计算图
    #      常驻显存，已改为 float(loss.detach())），另一个是 batch=256 的单批激活。
    #    - 表格数据集（ADULT/COMPAS/DRUG/DUTCH）与文本数据集显存占用低（数 GiB 级），
    #      不构成压力；高危场景集中在图像 + batch=256。
    # 2) detach 修复后仍可能 OOM，但根因常是「同卡邻居挤压」而非本进程：
    #    OOM 报错里的 "this process has X GiB in use" 是本进程实测占用（曾观测到
    #    仅 3~6 GiB 仍然 OOM），旁边 "Process NNN has 41.09 GiB" 才是真凶
    #    （可能是同卡的 FedFair，也可能是常驻的 VLLM::Worker 服务，后者占约 42 GiB）。
    #    排障顺序：先 nvidia-smi 看同卡还有谁 + 按「同卡峰值之和」排卡，再怀疑本算法。
    accumulation_steps = max(1, int(256 / param_dict['batch_size']))
    micro_batch_size = max(1, param_dict['batch_size'] // accumulation_steps)
    fair_sampler = construct_the_fairsampler(param_dict, model, client_i_fairbatch_dataset,
                                             micro_batch_size, 0.005, 'eqopp')

    if global_lb1 is not None:
        fair_sampler.lb1 = global_lb1
    if global_lb2 is not None:
        fair_sampler.lb2 = global_lb2

    client_i_dataloader = torch.utils.data.DataLoader(client_i_fairbatch_dataset, sampler=fair_sampler, num_workers=0)

    for epoch in range(algorithm_epoch_T):
        epoch_total_loss = 0
        epoch_total_size = 0
        batch_id = -1

        gpu_start_time = time.time()

        for batch in client_i_dataloader:
            batch_id += 1
            labels = batch[1][0].to(device)

            true_batch_size = labels.size()[0]
            epoch_total_size += true_batch_size

            with autocast_context(device, use_amp):
                if "SENT_CLF" in param_dict["task"]:
                    input_ids = batch[0][0, :, 0].to(device)
                    attention_mask = batch[0][0, :, 1].to(device)
                    features, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    imgs = batch[0][0].to(device)
                    logits, features = model(imgs)
                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch[0][0].to(device)
                    if len(X.shape) == 3 and X.shape[1] == 1:
                        X = X.squeeze(1)
                    logits, features = model(X)

                if "SENT_CLF" in param_dict["task"]:
                    activated_preds = logits
                    _, preds = torch.max(activated_preds, dim=1)
                    batch_loss = criterion(activated_preds, labels)
                elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                    activated_preds = logits[:, 0]
                    preds = (torch.sigmoid(activated_preds) >= 0.5).long()
                    batch_loss = criterion(activated_preds, labels.float())
                elif "Tabular_CLF" in param_dict["task"]:
                    activated_preds = logits[:, 0]
                    preds = (torch.sigmoid(activated_preds) >= 0.5).long()
                    batch_loss = criterion(activated_preds, labels.float())
                loss = torch.sum(batch_loss)
            if loss.item() != 0:
                # 梯度累积：损失除以 accumulation_steps 后反传，累计满一组再 step，
                # 使参数更新等价于在合并后的 256 样本大批上做一次平均损失更新。
                loss = loss / true_batch_size
                scale_backward(loss / accumulation_steps, scaler)

            if (batch_id + 1) % accumulation_steps == 0:
                scaler_step(scaler, optimizer)
                model.zero_grad()

            # 只累加标量（detach 到 Python float）。若直接 += 张量，整个 epoch 所有
            # 批次的运算图会被 epoch_total_loss 一直持有到 epoch 结束（虽然每批已
            # 反传，但引用未释放，显存不会回收），图像 batch=256/FP32 下实测单进程
            # 占用 37GB+，进而挤爆同卡其它作业。该变量仅用于日志，标量足够。
            # 除以 accumulation_steps 以保持与原 batch=256 口径同一量纲：原实现记录
            # Σ_b(mean_256)/N = mean/256，micro-batch 口径下 Σ_b(mean_64)/N = mean/64，
            # 乘 1/acc=1/4 后回到 mean/256，使日志数值可与旧结果/其它算法直接比较。
            epoch_total_loss += float(loss.detach()) / accumulation_steps

            if "SENT_CLF" in param_dict["task"]:
                del input_ids, attention_mask, labels
            elif "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                del imgs, labels
            elif "Tabular_CLF" in param_dict["task"]:
                del X, labels
            gc.collect()

        # 收尾：micro-batch 数不是 accumulation_steps 整数倍时，把残余梯度 step 掉，
        # 否则最后一个不满组的梯度会被丢弃，参数更新数比原实现少一次。
        if (batch_id + 1) % accumulation_steps != 0:
            scaler_step(scaler, optimizer)
            model.zero_grad()

        gpu_end_time = time.time()

        epoch_total_size = max(epoch_total_size, client_datasets_size_list[client_id])

        average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
        logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                    f"Client: {client_id} / {num_clients_K}; "
                    f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

    client_model_path = os.path.join(basic_path, "client_" + str(client_id + 1), 'model.pt')
    torch.save(model.cpu(), client_model_path)

    lb1 = fair_sampler.lb1
    lb2 = fair_sampler.lb2

    del model, fair_sampler
    gc.collect()
    torch.cuda.empty_cache()

    return {'lb1': lb1, 'lb2': lb2, 'gpu_seconds': gpu_end_time - gpu_start_time}


def FL_FairBatch(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len,
            start_round=0,
            resume_state=None
            ):
    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    # AMP 初始化
    use_amp = param_dict.get('use_amp', False)
    scaler = get_scaler(device, use_amp)
    # 断点续跑：恢复 AMP 梯度缩放器状态，使自动混合精度的缩放因子在续跑后保持连续
    # （与 FederatedAverage / PraFFL / FedFACT 的约定一致）。
    if resume_state is not None and resume_state.amp_scaler_state is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains AMP scaler state but AMP is disabled")
        scaler.load_state_dict(resume_state.amp_scaler_state)

    # 用入口解析好的 model_path，避免按 algorithm 名重拼 save_path 时丢掉
    # -variant_tag / -dp_epsilon 的路径隔离（同 FedFB）。
    basic_path = param_dict['model_path']

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"] or "IMG_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "Tabular_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # FairBatch 的 lambda 值需要在轮次间持久化
    global_lb1 = None
    global_lb2 = None
    # 断点续跑：恢复 FairBatch 的自适应 lambda（lb1/lb2）。save_checkpoint 以
    # extra_state 持久化这两个值；若不复原，续跑时 fair_sampler 会从 None 重新
    # 自适应，丢失此前的公平性调整进度。
    if resume_state is not None:
        saved_algo_state = resume_state.algorithm_state or {}
        global_lb1 = saved_algo_state.get('global_lb1', None)
        global_lb2 = saved_algo_state.get('global_lb2', None)

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    parallel_executor = ClientParallelExecutor(device=device, global_model=global_model, param_dict=param_dict, needs_global_model_during_training=False)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

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
        client_results = parallel_executor.run_clients(
            idxs_users, _train_single_client_fairbatch,
            param_dict=param_dict,
            training_dataloaders=training_dataloaders,
            algorithm_epoch_T=algorithm_epoch_T,
            client_datasets_size_list=client_datasets_size_list,
            num_clients_K=num_clients_K,
            basic_path=basic_path,
            iter_t=iter_t,
            communication_round_I=communication_round_I,
            client_dataset_list=client_dataset_list,
            global_lb1=global_lb1, global_lb2=global_lb2,
            use_amp=use_amp, scaler=scaler, criterion=criterion,
        )
        for i, result in enumerate(client_results):
            id = idxs_users[i]
            users_gpu_seconds_list[id] += result['gpu_seconds']
            global_lb1 = result['lb1']
            global_lb2 = result['lb2']

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # ── 收集客户端模型更新（用于梯度监控）──
        pre_agg_params = get_parameters(global_model)
        client_model_updates = []

        # Global operation
        logger.info("Parameter aggregation")
        client_states = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            client_params = get_parameters(selected_model)
            client_states.append(selected_model.state_dict())

            # 计算该客户端的更新量
            updates = {}
            for j, (p_local, p_global) in enumerate(zip(client_params, pre_agg_params)):
                updates[str(j)] = torch.tensor(p_local) - torch.tensor(p_global)
            client_model_updates.append(updates)

            del selected_model
            gc.collect()

        aggregated_state = average_state_dicts(client_states, weights=[client_datasets_size_list[j] for j in idxs_users])
        logger.info("Update Global Model")
        global_model.load_state_dict(aggregated_state, strict=True)
        del client_states, aggregated_state

        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        gc.collect()

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I'] and (iter_t + 1) % param_dict.get('test_eval_freq', 1) == 0:
            # 按任务类型选择评估函数：文本任务走 input_ids 版本，图像/表格任务
            # 的 batch 不含 input_ids，必须使用各自的分支，否则会 KeyError。
            if "IMG_CLF" in param_dict["task"] or "AUD_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
            elif "Tabular_CLF" in param_dict["task"]:
                evaluation = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
            else:
                evaluation = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
            accuracy, DEO, SPD, ACC_gap = evaluation[0], evaluation[1], evaluation[2], evaluation.acc_gap
            logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}, ACC_gap: {round(float(ACC_gap), 3)}")
            # ===== TensorBoard logging =====
            log_test_metrics(
                ACC_gap=float(ACC_gap), accuracy=float(accuracy), DEO=float(DEO), SPD=float(SPD),
                step=iter_t+1, gpu_seconds=total_gpu_seconds, avg_gpu_seconds=avg_gpu_seconds,
                communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size
            )
            log_system_metrics(step=iter_t+1, gpu_seconds=total_gpu_seconds, 
                               communication_cost=(iter_t + 1) * len(idxs_users) * 2 * model_MB_size,
                               selected_client_count=len(idxs_users), selected_clients=idxs_users.tolist(), model_mb_size=model_MB_size)
            flush()


        # ===== 深度监控（每轮都执行，包括最后一轮）=====
        cfg_deep = get_monitoring_config(param_dict)
        log_deep_metrics(global_model, param_dict, testing_dataloader, 
                         iter_t + 1, client_model_updates=client_model_updates)

        # 保存检查点（按 checkpoint_save_freq 间隔，含 FairBatch lambda 值）
        if param_dict.get('checkpoint_save_freq', 1) > 0 and iter_t % param_dict.get('checkpoint_save_freq', 1) == 0:
            save_checkpoint(
                param_dict=param_dict,
                iter_t=iter_t,
                global_model=global_model,
                total_gpu_seconds=total_gpu_seconds,
                client_selection_history=[idxs_users.tolist()] if hasattr(idxs_users, 'tolist') else [idxs_users],
                start_time=start_time,
                extra_state={
                    'global_lb1': global_lb1,
                    'global_lb2': global_lb2
                }
            )

            # 清理旧检查点，保留最近 N 个
            clean_old_checkpoints(param_dict, keep_latest=param_dict.get('checkpoint_keep_latest', 5))

    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FL_FairBatch.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
