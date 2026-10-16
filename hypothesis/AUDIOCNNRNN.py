# -*- coding: utf-8 -*-
"""语音分类模型：卷积-循环网络 / Audio classifier: Conv-RNN.

核心思想 / Core idea
-------------------
把一段音频的 log-Mel 频谱看作 ``[T, F]`` 的时序信号（T 为帧数、F 为梅尔频带数）：

1. **卷积层建模局部时频模式**：两层一维卷积沿时间轴滑动，每层后接 2 倍最大
   池化，使帧率逐步降为 1/4，从而在局部窗口内捕捉共振峰与能量突变；
2. **双向 GRU 建模长时依赖**：把卷积输出的每一帧当作序列元素送入双向
   GRU，同时利用前后文信息（语音判别性线索往往跨越数百毫秒）；
3. **时间均值池化得到定长表示**：对 GRU 输出在时间维取均值，使表示对片段
   长度不敏感，再用线性层输出分类 logit。

该结构对齐 FedAudio 基准中的 ``audio_conv_rnn``：卷积负责局部时频特征，
循环网络负责时序聚合，均值池化负责长度归一。

原始论文 / Reference
--------------------
FedAudio: A Federated Learning Benchmark for Audio and Speech Tasks.
Tuo Zhang, Tiantian Feng, Samiul Alam, Sunwoo Lee, Mi Zhang,
Shrikanth S. Narayanan, Salman Avestimehr. ICASSP 2023.
- 论文链接: https://arxiv.org/abs/2210.15707
- 实现参考: https://github.com/zhang-tuo-pdf/FedAudio/blob/main/model/conv_model.py

与上游实现的差异 / Deviations from upstream
------------------------------------------
1. 上游把 dropout 同时传给单层 GRU（单层 GRU 的 dropout 参数无效且会触发
   PyTorch 警告），此处只在卷积后保留显式 Dropout，GRU 不设 dropout。
2. 上游 ``pred_layer`` 输出 ``label_size`` 个 logits；本仓库统一采用"共享基座
   + 二分类头"的算法接口约定（``shared_base`` / ``out_layer``，前向返回
   ``(pred, feature)``），因此 ``out_layer`` 输出维度为 1，配合 ``torch.sigmoid``
   与 BCE 损失使用。
"""

import torch
import torch.nn as nn


class _AudioConvRNNBackbone(nn.Module):
    """log-Mel 频谱 ``[B, T, F]`` -> 定长表示 ``[B, 2 * hidden_size]`` 的卷积-循环基座。"""

    def __init__(self, feature_size, hidden_size=128, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        # 卷积沿时间轴滑动：输入 [B, F, T]，输出 [B, hidden_size, T / 4]
        self.conv = nn.Sequential(
            nn.Conv1d(feature_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # x: [B, T, F]，时间在前、特征在后（与 FedAudio 的 audio_conv_rnn 输入一致）
        hidden = self.conv(x.transpose(1, 2))          # [B, hidden_size, T / 4]
        hidden = hidden.transpose(1, 2).contiguous()   # [B, T / 4, hidden_size]
        output, _ = self.rnn(hidden)                   # [B, T / 4, 2 * hidden_size]
        return output.mean(dim=1)                      # [B, 2 * hidden_size]


class AudioConvRNN(nn.Module):
    """AUD_CLF 的语音分类器（Conv-RNN，二分类头）。"""

    def __init__(self, feature_size, hidden_size=128, dropout=0.2):
        super().__init__()
        self.shared_base = _AudioConvRNNBackbone(feature_size, hidden_size, dropout)
        self.out_layer = nn.Linear(hidden_size * 2, 1)

        # 为 scaffold 等算法提供的控制变量
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

    def forward(self, x, task=0, return_representation=True, return_logit=False):
        feature = self.shared_base(x)
        logit = self.out_layer(feature)
        pred = torch.sigmoid(logit)

        if return_logit:
            if return_representation:
                return logit, pred, feature
            return logit, pred
        if return_representation:
            return pred, feature
        return pred

    def only_backbone_forward(self, x):
        # 防御性转换：对齐主干首层权重 dtype（防止 Double/Half 输入导致卷积崩溃）
        first_layer = self.shared_base.conv[0]
        if x.dtype != first_layer.weight.dtype:
            x = x.to(first_layer.weight.dtype)
        return self.shared_base(x)

    def only_clf_forward(self, feature):
        # 防御性转换：feature 可能来自 AMP autocast 路径（Half），out_layer 权重为 Float32
        layer_dtype = self.out_layer.weight.dtype
        if feature.dtype != layer_dtype:
            feature = feature.to(layer_dtype)
        pred = torch.sigmoid(self.out_layer(feature))
        return feature, pred

    def shared_parameters(self):
        return (p for p in self.shared_base.parameters())

    def task_specific_parameters(self):
        return [p for p in self.out_layer.parameters()]

    def last_shared_parameters(self):
        return []
