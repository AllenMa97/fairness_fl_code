import torch
import torch.nn as nn

# 参考自：（ICML2024） Fair Resource Allocation in Multi-Task Learning

# We employ a networkcontaining a 9-layer convolutional neural network (CNN)as the backbone and a specific linear layer for each task.

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(ConditionalBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # 检查 batch size 是否为 1
        if x.size(0) == 1 and self.training:
            # 训练模式且 batch size 为 1 时，跳过 BN 层
            return x
        else:
            # 否则正常应用 BN 层
            return self.bn(x)

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ConditionalBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        # 检查 batch size 是否为 1（x.shape[0] 是 batch 维度）
        if x.size(0) == 1 and self.training:
            # 训练模式且 batch size 为 1 时，跳过 BN 层
            return x
        else:
            # 否则正常应用 BN 层
            return self.bn(x)


class RegularCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_base = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            ConditionalBatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            ConditionalBatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            ConditionalBatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            ConditionalBatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            ConditionalBatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            ConditionalBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            ConditionalBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(512, 512, bias=False),
            # nn.BatchNorm1d(512),
            ConditionalBatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            # nn.BatchNorm1d(512),
            ConditionalBatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # Prediction head
        # self.out_layer = nn.ModuleList([nn.Linear(512, 1) for _ in range(40)]) # 如果是40个任务的Multi-task Learning则用这个。但是注意可能会出现set_parameter的问题。
        self.out_layer = nn.Linear(512, 1)

        # 为scaffold算法提供的控制变量
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

    def forward(self, x, task=0, return_representation=True, return_logit=False):
        feature = self.shared_base(x)
        # 如果是Multi-task Learning则用这个
        # if task is None:
        #     pred = [torch.sigmoid(self.out_layer[task](feature)) for task in range(40)]
        # else:
        #     pred = torch.sigmoid(self.out_layer[task](feature))

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
        # 如果是Multi-task Learning则用这个
        # if task is None:
        #     pred = [torch.sigmoid(self.out_layer[task](feature)) for task in range(40)]
        # else:
        #     pred = torch.sigmoid(self.out_layer[task](feature))

        return feature

    def only_clf_forward(self, feature):
        pred = torch.sigmoid(self.out_layer(feature))
        return feature, pred

    def shared_parameters(self):
        return (p for p in self.shared_base.parameters())

    def task_specific_parameters(self):
        return_list = []
        for task in range(40):
            return_list += [p for p in self.out_layer[task].parameters()]
        return return_list

    def last_shared_parameters(self):
        return []
