import numpy as np
import torch
from torch import nn

class RegularLogisticRegression(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(RegularLogisticRegression, self).__init__()
        self.layer = nn.Linear(input_size, output_size, bias=False)  # No bias them in LogisticRegression case
        self.sigmoid = nn.Sigmoid()
        # torch.nn.init.zeros_(self.layer.weight)  # The parameter is initialized to zero (Following the setting in Renyi)

        # 为scaffold算法提供的控制变量
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

        # 为ditto算法创建的delta_global_model
        self.delta_global_model = {}

    def forward(self, X):
        try:
            out = self.layer(X)
        except:
            out = self.layer(X.float())
        out = self.sigmoid(out)
        return out
#
# class FedFBLogisticRegression(nn.Module):
#     def __init__(self, input_size, output_size=2):
#         super(FedFBLogisticRegression, self).__init__()
#         self.layer = nn.Linear(input_size, output_size, bias=False)  # No bias them in LogisticRegression case
#         self.sigmoid = nn.Sigmoid()
#         torch.nn.init.zeros_(self.layer.weight)  # The parameter is initialized to zero (Following the setting in Renyi)
#
#     def forward(self, X):
#         try:
#             logits = self.layer(X)
#         except:
#             logits = self.layer(X.float())
#         probas = self.sigmoid(logits)
#
#         return probas.type(torch.FloatTensor), logits
