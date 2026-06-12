import torch
import torch.nn as nn

class RegularANN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()

        self.shared_base = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.Linear(input_size * 2, input_size),
        )
        # Prediction head
        self.out_layer = nn.Linear(input_size, output_size)

        # 为scaffold算法提供的控制变量
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

    def forward(self, x, return_representation=True, return_logit=False):
        try:
            feature = self.shared_base(x)
        except:
            feature = self.shared_base(x.float())

        logit = self.out_layer(feature)
        pred = torch.sigmoid(self.out_layer(feature))

        if return_logit:
            if return_representation:
                return logit, pred, feature
            else:
                return logit, pred
        else:
            if return_representation:
                return pred, feature
            else:
                return pred

    def only_backbone_forward(self, x):
        feature = self.shared_base(x)
        return feature

    def only_clf_forward(self, feature):
        pred = torch.sigmoid(self.out_layer(feature))
        return feature, pred

