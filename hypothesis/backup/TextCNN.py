import torch
import torch.nn.functional as F
from torch import nn

class TextCNN(nn.Module):
    def __init__(self, num_classes, input_size, filter_sizes_list=[2, 3, 4, 5], num_filters=4, dropout_rate=0.2):
        super(TextCNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Define the hyperparameters
        self.filter_sizes = filter_sizes_list
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.input_size)) for K in self.filter_sizes]
        )
        self.block = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        tokens = conv(tokens)
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)
        tokens = F.max_pool1d(tokens, tokens.size(2))
        out = tokens.squeeze(2)
        return out

    def forward(self, feature):
        tokens = feature.unsqueeze(1)
        out = torch.cat([self.conv_pool(tokens, conv) for conv in self.convs], 1)
        predicts = self.block(out)
        return predicts

