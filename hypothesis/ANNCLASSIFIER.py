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
        # 防御性转换：部分算法会把 numpy(float64) 的原型/质心直接转成 Tensor 传入，
        # 这里对齐主干首层权重的 dtype，避免 Double vs Float 的 matmul 崩溃。
        first_layer = self.shared_base[0]
        if isinstance(first_layer, (nn.Linear, nn.Conv2d)) and x.dtype != first_layer.weight.dtype:
            x = x.to(first_layer.weight.dtype)
        feature = self.shared_base(x)
        return feature

    def only_clf_forward(self, feature):
        # 防御性转换：feature 可能来自 AMP autocast 路径（Half / BFloat16），
        # 而 out_layer 权重保持 Float32；显式对齐 dtype 避免 matmul 崩溃。
        layer_dtype = self.out_layer.weight.dtype
        if feature.dtype != layer_dtype:
            feature = feature.to(layer_dtype)
        pred = torch.sigmoid(self.out_layer(feature))
        return feature, pred

