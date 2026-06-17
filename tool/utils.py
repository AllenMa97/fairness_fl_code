#!/usr/bin/python
# -*- coding: utf-8 -*-
import gc
import re
import os
# import thop
import torch
import numpy as np
import scipy
import pandas as pd
import time
import math
import random
import argparse
import statistics
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from scipy.stats import beta as beta_distribution
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List
from collections import OrderedDict
from tool.logger import *
from tool.amp_utils import autocast_context

sys.setrecursionlimit(10000)


def get_HM_by_two_value(acc, FR):
    HM = statistics.harmonic_mean([float(acc), float(FR)])
    return HM


# 常见的预训练语言模型 embedding 维度查找表
# Common pretrained language model embedding dimension lookup table
_EMB_DIM_LOOKUP = {
    # BERT 系列
    'bert': 768,
    'roberta': 768,
    'distilbert': 768,
    'albert': 768,
    'electra': 768,
    'deberta': 768,
    'xlnet': 768,
    # GPT 系列
    'gpt2': 768,
    'gpt_neo': 768,
    'gpt_neox': 768,
    # T5 系列
    't5': 512,
    # LLaMA/Mistral 系列
    'llama': 4096,
    'mistral': 4096,
    'qwen': 4096,
    'gemma': 2048,
    # 其他常见
    'bge': 768,
    'e5': 768,
    'roformer': 768,
}


def get_emb_dim(param_dict=None, model=None, default=768):
    """获取 embedding 维度，按优先级依次尝试多种方式，保证不会报错。

    优先级：
        1. param_dict['emb_dim']                          —— 实验配置中显式设置
        2. 从模型结构中推断                                    —— 检查 bert.config / shared_base.Linear.in_features
        3. 查表匹配常见预训练模型的 hidden_size                   —— 基于模型 class name 匹配
        4. 返回 default（默认 768）                            —— 最终兜底

    Args:
        param_dict: 实验参数字典，可能包含 'emb_dim' 键
        model:      torch.nn.Module 模型实例，用于推断维度
        default:    最终兜底值

    Returns:
        int: embedding 维度
    """
    # ---- 1) param_dict 中的显式配置 ----
    if param_dict is not None:
        emb_dim = param_dict.get('emb_dim', None)
        if emb_dim is not None:
            return emb_dim

    # ---- 2) 从模型结构推断 ----
    if model is not None:
        # 2a) BERT / HuggingFace transformer 系列 —— 有 bert / roberta / ... 属性
        for attr_name in ('bert', 'roberta', 'distilbert', 'albert', 'electra',
                           'deberta', 'xlnet', 'transformer', 'encoder'):
            if hasattr(model, attr_name):
                encoder = getattr(model, attr_name)
                if hasattr(encoder, 'config') and hasattr(encoder.config, 'hidden_size'):
                    return encoder.config.hidden_size
                # 某些包装过的 BERT（例如 model.bert 是 BertModel）
                if hasattr(encoder, 'embeddings'):
                    try:
                        # word_embeddings.weight.shape = [vocab_size, hidden_size]
                        return encoder.embeddings.word_embeddings.weight.shape[1]
                    except Exception:
                        pass

        # 2b) 通用 nn.Module —— 检查 shared_base / backbone 的第一个 Linear 层
        for backbone_attr in ('shared_base', 'backbone', 'encoder', 'features'):
            if hasattr(model, backbone_attr):
                backbone = getattr(model, backbone_attr)
                # 递归查找第一个 nn.Linear 的 in_features
                if isinstance(backbone, torch.nn.Sequential):
                    for layer in backbone:
                        if isinstance(layer, torch.nn.Linear):
                            return layer.in_features
                elif isinstance(backbone, torch.nn.Linear):
                    return backbone.in_features

        # 2c) 直接检查模型自身的 state_dict —— 找 word_embeddings
        try:
            state_dict = model.state_dict()
            for key, tensor in state_dict.items():
                if 'word_embeddings' in key and 'weight' in key:
                    return tensor.shape[1]
        except Exception:
            pass

    # ---- 3) 查表匹配常见模型名 ----
    if model is not None:
        model_class_name = type(model).__name__.lower()
        for keyword, dim in _EMB_DIM_LOOKUP.items():
            if keyword in model_class_name:
                return dim

    # ---- 4) 最终兜底 ----
    return default

def get_specific_time():
    now = time.localtime()
    year, month, day = str(now.tm_year), str(now.tm_mon), str(now.tm_mday)
    hour, minute, second = str(now.tm_hour), str(now.tm_min), str(now.tm_sec)
    return str(year + "_" + month + "_" + day + "_" + hour + "h" + minute + "m" + second + "s")


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    x = x.lower()
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def check_and_make_the_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# compute the cos similarity between a and b. a, b are numpy arrays
def cos_sim(a, b):
    return 1 - scipy.spatial.distance.cosine(a, b)


def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        print("match_true:", match_true.data, " ;pred:", pred.data, " ;true:", true.data, " ;match:", match.data,
              " ;total:", total)
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def normalization(x):
    """"
    归一化到区间{0,1]
    返回副本
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def get_tensor_parameters(net) -> List[torch.Tensor]:
    return list(net.parameters())


def set_parameters(net, parameters: List[np.ndarray]):
    # 检查parameters的结果是否存在非np.ndarray的项，如有则转换
    checked_parameters = []
    for item in parameters:
        if isinstance(item, np.ndarray):
            checked_parameters.append(item)
        elif isinstance(item, np.float64):
            checked_parameters.append(np.array([item]))
    if len(checked_parameters) != 0:
        parameters = checked_parameters
    else:
        logger.warning("The checked_parameters is empty, use the not checked (origin) parameters!")

    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def save_model(param_dict, updated_global_model, client_model_list, iter_t, optim):
    logger.info("Communication Round %d Global Models Saving" % (iter_t + 1))
    # TODO start
    model_state_dict = updated_global_model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        # 'generator': generator_state_dict,
        'opt': param_dict,
        'optims': optim,
    }
    check_and_make_the_path(param_dict['model_path'])
    torch.save(checkpoint, os.path.join(param_dict['model_path'], "step_%d_" % iter_t + "global_model.pt"))
    # TODO end
    # torch.save(updated_global_model, os.path.join(param_dict['model_path'], "step_%d_" % iter_t + "global_model.pkl"))
    logger.info("Communication Round %d Client Models Saving" % (iter_t + 1))
    for client_id, client_model in enumerate(client_model_list):
        _ = os.path.join(param_dict['model_path'],
                         "client_" + str(client_id + 1), "step_%d_" % iter_t + "model.pkl")
        check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
        torch.save(client_model, _)


def save_model_sepa(param_dict, client_model_list, epoch):
    check_and_make_the_path(param_dict['model_path'])
    logger.info("Total Epoch %d Separate Client Models Saving" % (epoch))
    for client_id, client_model in enumerate(client_model_list):
        _ = os.path.join(param_dict['model_path'],
                         "client_" + str(client_id + 1), "step_%d_" % epoch + "model.pkl")
        check_and_make_the_path(os.path.join(param_dict['model_path'], "client_" + str(client_id + 1)))
        torch.save(client_model, _)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# [UNUSED] get_all_Fairness_score - 当前未被任何地方调用，已注释。
# 实际使用的是下方的 get_Fairness_score 函数。
# This function is currently unused. The active function is get_Fairness_score below.
#
# 原始版本存在 DEO 定义错误（EOP 在二分类下恒为0，因为 P(pred=1|A,Y=1) + P(pred=0|A,Y=1) = 1）。
# 修复后的正确逻辑如下（保留供参考）：
#
# def get_all_Fairness_score(preds, sa, labels):
#     """计算多种公平性指标: DP, EOD, EOP, DEO, SPD"""
#     Pr_pred1_A1_Y1, Pr_pred1_A0_Y1 = 0, 0
#     Pr_pred1_A1_Y0, Pr_pred1_A0_Y0 = 0, 0
#     Pr_pred1_A1, Pr_pred1_A0 = 0, 0
#     Pr_A1_Y1, Pr_A0_Y1 = 0, 0
#     Pr_A1_Y0, Pr_A0_Y0 = 0, 0
#     Pr_A1, Pr_A0 = 0, 0
#
#     total = len(preds)
#     for i in range(total):
#         pred, A, Y = preds[i], sa[i], labels[i]
#         if A == 1:
#             Pr_A1 += 1 / total
#             if Y == 1:
#                 Pr_A1_Y1 += 1 / total
#                 if pred == 1:
#                     Pr_pred1_A1_Y1 += 1 / total
#             elif Y == 0:
#                 Pr_A1_Y0 += 1 / total
#                 if pred == 1:
#                     Pr_pred1_A1_Y0 += 1 / total
#             if pred == 1:
#                 Pr_pred1_A1 += 1 / total
#         else:
#             Pr_A0 += 1 / total
#             if Y == 1:
#                 Pr_A0_Y1 += 1 / total
#                 if pred == 1:
#                     Pr_pred1_A0_Y1 += 1 / total
#             elif Y == 0:
#                 Pr_A0_Y0 += 1 / total
#                 if pred == 1:
#                     Pr_pred1_A0_Y0 += 1 / total
#             if pred == 1:
#                 Pr_pred1_A0 += 1 / total
#
#     # DEO: |P(pred=1|A=0,Y=1) - P(pred=1|A=1,Y=1)| (TPR差异)
#     try:
#         Pr_pred1_given_A0_Y1 = Pr_pred1_A0_Y1 / Pr_A0_Y1
#     except Exception:
#         Pr_pred1_given_A0_Y1 = 0
#     try:
#         Pr_pred1_given_A1_Y1 = Pr_pred1_A1_Y1 / Pr_A1_Y1
#     except Exception:
#         Pr_pred1_given_A1_Y1 = 0
#     DEO = abs(Pr_pred1_given_A0_Y1 - Pr_pred1_given_A1_Y1)
#
#     # SPD: P(pred=1|A=0) - P(pred=1|A=1)
#     try:
#         Pr_pred1_given_A0 = Pr_pred1_A0 / Pr_A0
#     except Exception:
#         Pr_pred1_given_A0 = 0
#     try:
#         Pr_pred1_given_A1 = Pr_pred1_A1 / Pr_A1
#     except Exception:
#         Pr_pred1_given_A1 = 0
#     SPD = Pr_pred1_given_A0 - Pr_pred1_given_A1
#
#     # DP: |P(pred=1|A=0) - P(pred=1|A=1)| (与SPD相同，取绝对值)
#     DP = abs(SPD)
#
#     # EOD: |P(pred=1|A=0,Y=0) - P(pred=1|A=1,Y=0)| + |P(pred=1|A=0,Y=1) - P(pred=1|A=1,Y=1)|
#     try:
#         Pr_pred1_given_A0_Y0 = Pr_pred1_A0_Y0 / Pr_A0_Y0
#     except Exception:
#         Pr_pred1_given_A0_Y0 = 0
#     try:
#         Pr_pred1_given_A1_Y0 = Pr_pred1_A1_Y0 / Pr_A1_Y0
#     except Exception:
#         Pr_pred1_given_A1_Y0 = 0
#     EOD = abs(Pr_pred1_given_A0_Y0 - Pr_pred1_given_A1_Y0) + DEO
#
#     # EOP: P(pred=1|A=0,Y=1) - P(pred=1|A=1,Y=1) (不取绝对值)
#     EOP = Pr_pred1_given_A0_Y1 - Pr_pred1_given_A1_Y1
#
#     return DP, EOD, EOP, DEO, SPD
# # END [UNUSED] get_all_Fairness_score




def get_Fairness_score(preds, sa, labels):
    Pr_pred1_A1_Y1, Pr_pred1_A0_Y1, Pr_pred0_A1_Y1, Pr_pred0_A0_Y1 = 0, 0, 0, 0  # DEO相关的联合概率
    Pr_pred1_given_A1_Y1, Pr_pred1_given_A0_Y1, Pr_pred0_given_A1_Y1, Pr_pred0_given_A0_Y1 = 0, 0, 0, 0  # DEO相关的条件概率
    Pr_A1_Y1, Pr_A0_Y1 = 0, 0  # DEO相关的概率

    Pr_pred1_A0, Pr_pred1_A1 = 0, 0  # SPD相关的联合概率
    Pr_pred1_given_A0, Pr_pred1_given_A1 = 0, 0  # SPD相关的条件概率
    Pr_A0, Pr_A1 = 0, 0  # SPD相关的概率

    total = len(preds)  # 总数
    for i in range(total):
        pred, A, Y = preds[i], sa[i], labels[i]
        if A == 1:
            Pr_A1 += 1 / total
            if Y == 1:
                Pr_A1_Y1 += 1 / total
                if pred == 1:
                    Pr_pred1_A1_Y1 += 1 / total
                else:
                    Pr_pred0_A1_Y1 += 1 / total

            if pred == 1:
                Pr_pred1_A1 += 1 / total

        else:
            Pr_A0 += 1 / total
            if Y == 1:
                Pr_A0_Y1 += 1 / total
                if pred == 1:
                    Pr_pred1_A0_Y1 += 1 / total
                else:
                    Pr_pred0_A0_Y1 += 1 / total

            if pred == 1:
                Pr_pred1_A0 += 1 / total
    try:
        Pr_pred1_given_A1_Y1 += Pr_pred1_A1_Y1 / Pr_A1_Y1
    except Exception:
        Pr_pred1_given_A1_Y1 += 0
    try:
        Pr_pred0_given_A1_Y1 += Pr_pred0_A1_Y1 / Pr_A1_Y1
    except Exception:
        Pr_pred0_given_A1_Y1 += 0
    try:
        Pr_pred1_given_A0_Y1 += Pr_pred1_A0_Y1 / Pr_A0_Y1
    except Exception:
        Pr_pred1_given_A0_Y1 += 0
    try:
        Pr_pred0_given_A0_Y1 += Pr_pred0_A0_Y1 / Pr_A0_Y1
    except Exception:
        Pr_pred0_given_A0_Y1 += 0

    DEO = abs(Pr_pred1_given_A0_Y1 - Pr_pred1_given_A1_Y1)

    try:
        Pr_pred1_given_A0 += Pr_pred1_A0 / Pr_A0
    except Exception:
        Pr_pred1_given_A0 += 0
    try:
        Pr_pred1_given_A1 += Pr_pred1_A1 / Pr_A1
    except Exception:
        Pr_pred1_given_A1 += 0

    SPD = Pr_pred1_given_A0 - Pr_pred1_given_A1
    return DEO, SPD

# 测试 moji 数据集的公平性
def compute_moji_fairness_metrics(preds, labels, sa):
    """
    计算 Independence, Separation 和 Sufficiency 指标
    """
    df = pd.DataFrame({
        'preds': preds,
        'labels': labels,
        'sa': sa
    })

    # 独立性 (Independence): KL散度
    # 计算两组 (sa=0 和 sa=1) 的预测分布
    pred_dist_sa0 = df[df['sa'] == 0]['preds'].value_counts(normalize=True).sort_index()
    pred_dist_sa1 = df[df['sa'] == 1]['preds'].value_counts(normalize=True).sort_index()

    # 确保两个分布有相同的索引
    all_classes = sorted(df['preds'].unique())
    pred_dist_sa0 = pred_dist_sa0.reindex(all_classes, fill_value=1e-6)
    pred_dist_sa1 = pred_dist_sa1.reindex(all_classes, fill_value=1e-6)

    independence_kl = entropy(pred_dist_sa0, pred_dist_sa1)

    # 分离度 (Separation): TPR 和 FPR
    separation = {}
    groups = df['sa'].unique()
    binary_labels = df['labels'].nunique() == 2
    if binary_labels:
        for group in groups:
            group_df = df[df['sa'] == group]
            if len(group_df) > 0:
                tn, fp, fn, tp = confusion_matrix(group_df['labels'], group_df['preds'], labels=[0, 1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                separation[f'sa_{group}_TPR'] = tpr
                separation[f'sa_{group}_FPR'] = fpr

        # 计算不同组间的TPR和FPR差异
        print(separation)
        if 'sa_0_TPR' in separation and 'sa_1_TPR' in separation:
            tpr_diff = abs(separation['sa_0_TPR'] - separation['sa_1_TPR'])
        else:
            tpr_diff = None

        if 'sa_0_FPR' in separation and 'sa_1_FPR' in separation:
            fpr_diff = abs(separation['sa_0_FPR'] - separation['sa_1_FPR'])
        else:
            fpr_diff = None
    else:
        tpr_diff = None
        fpr_diff = None

    # 充分性 (Sufficiency): P(y | pred, sa)
    sufficiency = {}
    for pred in df['preds'].unique():
        for group in groups:
            subset = df[(df['preds'] == pred) & (df['sa'] == group)]
            if len(subset) > 0:
                p_y = subset['labels'].value_counts(normalize=True).to_dict()
                sufficiency[f'pred_{pred}_sa_{group}_P(y)'] = p_y

    # 计算每个预测类别下不同组的标签分布的KL散度
    sufficiency_kl = 0
    for pred in df['preds'].unique():
        p_y_sa0 = df[(df['preds'] == pred) & (df['sa'] == 0)]['labels'].value_counts(normalize=True).sort_index()
        p_y_sa1 = df[(df['preds'] == pred) & (df['sa'] == 1)]['labels'].value_counts(normalize=True).sort_index()

        # 确保两个分布有相同的索引
        all_labels = sorted(df['labels'].unique())
        p_y_sa0 = p_y_sa0.reindex(all_labels, fill_value=1e-6)
        p_y_sa1 = p_y_sa1.reindex(all_labels, fill_value=1e-6)

        sufficiency_kl += entropy(p_y_sa0, p_y_sa1)

    fairness_metrics = {
        'Independence_KL': independence_kl,
        'Separation_TPR_Diff': tpr_diff,
        'Separation_FPR_Diff': fpr_diff,
        'Sufficiency_KL': sufficiency_kl
    }

    return fairness_metrics

def test_moji(param_dict, testing_dataloader, n_examples):
    device = param_dict['device']
    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(param_dict["num_clients_K"]) + "Clients")
    # 读取持久化结果
    for k in range(param_dict["num_clients_K"]):
        torch.cuda.empty_cache()
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        testing_model = torch.load(full_path, weights_only=False)
        testing_model.eval()
        testing_model.zero_grad()
        testing_model.to(device)

        correct_predictions = 0

        all_preds = []
        all_labels = []
        all_sa = []
        # print(testing_dataloader)
        use_amp = param_dict.get('use_amp', False)
        with torch.no_grad():
            for index, d in enumerate(testing_dataloader):
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)
                sa = d["sa"].to(device)

                with autocast_context(device, use_amp):
                    _, logits = testing_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                activated_preds = logits.softmax(dim=1)
                _, preds = torch.max(activated_preds, dim=1)

                correct_predictions += torch.sum(preds == labels)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_sa.extend(sa.cpu().numpy())

        accuracy = correct_predictions.double() / n_examples

        # 公平性指标计算
        DEO, SPD = get_Fairness_score(all_preds, all_sa, all_labels)

        #
        # fairness_metrics = compute_moji_fairness_metrics(all_preds, all_labels, all_sa)
        # fairness_string = ""
        # for metric, value in fairness_metrics.items():
        #     fairness_string += f' {metric}: {value}\n'
        # logger.info(f'''
        # Fairness at
        # [[[
        # dataset {param_dict["dataset"]}
        # algorithm {param_dict["algorithm"]}
        # split_strategy {param_dict["split_strategy"]}
        # client {param_dict["num_clients_K"]}
        # batch_size {param_dict["batch_size"]}
        # algorithm_epoch_T {param_dict["algorithm_epoch_T"]}
        # communication_round_I {param_dict["communication_round_I"]}
        # ]]]
        # Accuracy: {accuracy}
        # Fairness Metrics:\n {fairness_string}
        # ''')

        # logger.info(f'''
        #         Fairness at
        #         [[[
        #         dataset {param_dict["dataset"]}
        #         algorithm {param_dict["algorithm"]}
        #         split_strategy {param_dict["split_strategy"]}
        #         client {param_dict["num_clients_K"]}
        #         algorithm_epoch_T {param_dict["algorithm_epoch_T"]}
        #         communication_round_I {param_dict["communication_round_I"]}
        #         ]]]
        #         Accuracy: {accuracy}
        #         DEO: {DEO}
        #         SPD: {SPD}
        #         ''')

    return accuracy, DEO, SPD

def compute_bios_fairness_metrics(preds, labels, protected):
    """
    计算多分类问题的 Independence, Separation 和 Sufficiency 指标
    """
    df = pd.DataFrame({
        'preds': preds,
        'labels': labels,
        'protected': protected
    })

    # 独立性 (Independence): KL散度
    pred_dist_protected0 = df[df['protected'] == 0]['preds'].value_counts(normalize=True).sort_index()
    pred_dist_protected1 = df[df['protected'] == 1]['preds'].value_counts(normalize=True).sort_index()

    all_classes = sorted(df['preds'].unique())
    pred_dist_protected0 = pred_dist_protected0.reindex(all_classes, fill_value=1e-6)
    pred_dist_protected1 = pred_dist_protected1.reindex(all_classes, fill_value=1e-6)
    independence_kl = entropy(pred_dist_protected0, pred_dist_protected1)

    # 分离度 (Separation): 多分类混淆矩阵
    separation = {}
    groups = df['protected'].unique()
    all_labels = sorted(df['labels'].unique())

    for group in groups:
        group_df = df[df['protected'] == group]
        if len(group_df) > 0:
            cm = confusion_matrix(group_df['labels'], group_df['preds'], labels=all_labels)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            separation[f'protected_{group}_confusion_matrix'] = cm_norm

    # 计算不同组间的混淆矩阵差异
    if all(f'protected_{group}_confusion_matrix' in separation for group in groups):
        cm_diff = np.abs(separation['protected_0_confusion_matrix'] - separation['protected_1_confusion_matrix'])
        separation['confusion_matrix_diff'] = cm_diff
    else:
        separation['confusion_matrix_diff'] = None

    # 充分性 (Sufficiency): P(y | pred, protected)
    sufficiency = {}
    sufficiency_kl = 0

    for pred in df['preds'].unique():
        for group in groups:
            subset = df[(df['preds'] == pred) & (df['protected'] == group)]
            if len(subset) > 0:
                p_y = subset['labels'].value_counts(normalize=True).sort_index()
                p_y = p_y.reindex(all_labels, fill_value=1e-6)
                sufficiency[f'pred_{pred}_protected_{group}_P(y)'] = p_y

        # 计算每个预测类别下不同组的标签分布的KL散度
        p_y_protected0 = sufficiency.get(f'pred_{pred}_protected_0_P(y)')
        p_y_protected1 = sufficiency.get(f'pred_{pred}_protected_1_P(y)')

        if p_y_protected0 is not None and p_y_protected1 is not None:
            sufficiency_kl += entropy(p_y_protected0, p_y_protected1)

    fairness_metrics = {
        'Independence_KL': independence_kl,
        'Separation_Confusion_Matrix_Diff': separation['confusion_matrix_diff'],
        'Sufficiency_KL': sufficiency_kl
    }

    return fairness_metrics

def test_bios(param_dict, testing_dataloader, n_examples):
    device = param_dict['device']
    basic_path = os.path.join("./save_path", param_dict['dataset_name'],
                              param_dict['split_strategy'],
                              param_dict['algorithm'],
                              param_dict['hypothesis'],
                              str(param_dict["num_clients_K"]) + "Clients")
    # 读取持久化结果
    for k in range(param_dict["num_clients_K"]):
        torch.cuda.empty_cache()
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        testing_model = torch.load(full_path, weights_only=False)
        testing_model.eval()
        testing_model.zero_grad()
        testing_model.to(device)

        correct_predictions = 0

        all_preds = []
        all_labels = []
        all_protected = []
        # print(testing_dataloader)
        with torch.no_grad():
            for d in tqdm(testing_dataloader, desc="Evaluating"):
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)
                protected = d["protected"].to(device)

                _, logits = testing_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                activated_preds = logits.softmax(dim=1)
                _, preds = torch.max(activated_preds, dim=1)

                correct_predictions += torch.sum(preds == labels)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_protected.extend(protected.cpu().numpy())

        accuracy = correct_predictions.double() / n_examples

        # 公平性指标计算
        fairness_metrics = compute_moji_fairness_metrics(all_preds, all_labels, all_protected)
        fairness_string = ""
        for metric, value in fairness_metrics.items():
            fairness_string += f' {metric}: {value}\n'

        logger.info(f'''
            Fairness at 
            [[[
            dataset {param_dict["dataset"]}
            algorithm {param_dict["algorithm"]}
            split_strategy {param_dict["split_strategy"]}
            client {param_dict["num_clients_K"]}
            batch_size {param_dict["batch_size"]} 
            algorithm_epoch_T {param_dict["algorithm_epoch_T"]}
            communication_round_I {param_dict["communication_round_I"]}
            ]]]
            Accuracy: {accuracy}
            Fairness Metrics:\n {fairness_string}
            ''')

def FL_fairness_and_accuracy_test(testing_model, param_dict, testing_dataloader, n_examples):
    device = param_dict['device']

    testing_model.eval()
    testing_model.zero_grad()
    testing_model.to(device)

    correct_predictions = 0

    all_preds = []
    all_labels = []
    all_sa = []
    # print(testing_dataloader)
    with torch.no_grad():
        for index, d in enumerate(testing_dataloader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            sa = d["protected"].to(device)

            _, logits = testing_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            activated_preds = logits.softmax(dim=1)
            _, preds = torch.max(activated_preds, dim=1)

            correct_predictions += torch.sum(preds == labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sa.extend(sa.cpu().numpy())

        accuracy = correct_predictions.double() / n_examples

        # 公平性指标计算
        DEO, SPD = get_Fairness_score(all_preds, all_sa, all_labels)

    return accuracy, DEO, SPD

def FL_fairness_and_accuracy_test_4_IMG_CLF(testing_model, param_dict, testing_dataloader, n_examples):
    device = param_dict['device']

    testing_model.eval()
    testing_model.zero_grad()
    testing_model.to(device)

    correct_predictions = 0

    all_preds = []
    all_labels = []
    all_sa = []
    # print(testing_dataloader)
    use_amp = param_dict.get('use_amp', False)
    with torch.no_grad():
        for index, d in enumerate(testing_dataloader):
            imgs = d["img"].to(device)
            labels = d["labels"].to(device)
            sa = d["protected"].to(device)

            with autocast_context(device, use_amp):
                tmp_preds, features = testing_model(imgs)
            preds = torch.where(tmp_preds > 0.5, 1, 0).float().squeeze(-1)
            correct_predictions += torch.sum(preds == labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sa.extend(sa.cpu().numpy())

        accuracy = correct_predictions.double() / n_examples

        # 公平性指标计算
        DEO, SPD = get_Fairness_score(all_preds, all_sa, all_labels)

    return accuracy, DEO, SPD


def FL_fairness_and_accuracy_test_4_Tabular_CLF(testing_model, param_dict, testing_dataloader, n_examples):
    device = param_dict['device']

    testing_model.eval()
    testing_model.zero_grad()
    testing_model.to(device)

    acc_numerator = 0
    acc_denominator = n_examples

    num_s1_pred1 = 0
    num_s1_pred0 = 0
    num_s0_pred1 = 0
    num_s0_pred0 = 0

    num_s1_pred1_y1 = 0
    num_s1_pred1_y0 = 0
    num_s0_pred1_y1 = 0
    num_s0_pred1_y0 = 0

    num_s1_y1 = 0
    num_s1_y0 = 0
    num_s0_y1 = 0
    num_s0_y0 = 0

    # Model testing
    testing_model.eval()
    use_amp = param_dict.get('use_amp', False)
    with torch.no_grad():
        for batch_index, batch in enumerate(testing_dataloader):
            try:
                X = batch["X"].to(device)
                y = batch["labels"].to(device)
                with autocast_context(device, use_amp):
                    if "ANN" in str(type(testing_model)):
                        tmp, __ = testing_model(X)
                    elif "LogisticRegression" in str(type(testing_model)):
                        tmp = testing_model(X)
                    else:
                        tmp = testing_model(X)
                prediction = (tmp >= 0.5).reshape(-1)
                acc_numerator += sum(prediction.eq(y))

                s = batch["protected"]

                y_0 = (y == 0).int().reshape(-1).to(device)
                y_1 = (y == 1).int().reshape(-1).to(device)
                s_1 = (s == 1).int().to(device)
                s_0 = (s == 0).int().to(device)
                pred_1 = (prediction == 1).int().to(device)
                pred_0 = (prediction == 0).int().to(device)

                num_s1_pred1 += (s_1 * pred_1).sum().to(device)
                num_s1_pred0 += (s_1 * pred_0).sum().to(device)
                num_s0_pred1 += (s_0 * pred_1).sum().to(device)
                num_s0_pred0 += (s_0 * pred_0).sum().to(device)

                num_s1_pred1_y1 += (s_1 * pred_1 * y_1).sum().to(device)
                num_s1_pred1_y0 += (s_1 * pred_1 * y_0).sum().to(device)
                num_s0_pred1_y1 += (s_0 * pred_1 * y_1).sum().to(device)
                num_s0_pred1_y0 += (s_0 * pred_1 * y_0).sum().to(device)

                num_s1_y1 += (s_1 * y_1).sum().to(device)
                num_s1_y0 += (s_1 * y_0).sum().to(device)
                num_s0_y1 += (s_0 * y_1).sum().to(device)
                num_s0_y0 += (s_0 * y_0).sum().to(device)

            except Exception as e:
                continue

    acc = acc_numerator / acc_denominator
    # logger.info(f"Testing model acc: {acc}")

    a = num_s0_pred1_y1 / num_s0_y1
    b = num_s1_pred1_y1 / num_s1_y1
    # logger.info(f"P(y^ = 1 | s = 0, y=1) = {a} , P(y^ = 1 | s = 1, y=1) = {b} ")

    # This definition is copy from Renyi
    DEO = abs( a - b )
    # logger.info(f"Difference of Equality of Opportunity violation (DEO): {DEO}")

    # This definition is copy from FairFed
    SPD = (num_s0_pred1/(num_s0_y1+num_s0_y0)) - (num_s1_pred1/(num_s1_y1+num_s1_y0))
    # logger.info(f"Statistical Parity Difference (SPD): {SPD}")

    return acc, DEO, SPD





def communication_cost_simulated_by_beta_distribution(client_number, alpha=0.3, beta=1):
    x = np.arange(0, 1, 1 / client_number)
    y = beta_distribution.pdf(x, alpha, beta)
    for index in range(len(y)):
        if math.isinf(y[index]):
            y[index] = 16
        elif math.isnan(y[index]):
            y[index] = 0.001
        else:
            y[index] = round(y[index], 4) + 1
    descending_order_list = [i for i in range(client_number)]
    random.shuffle(descending_order_list)
    return y, descending_order_list