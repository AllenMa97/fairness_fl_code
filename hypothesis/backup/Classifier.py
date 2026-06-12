import torch.nn as nn

from .TransformerBlock.TransformerEncoder import PositionalEncoding, TransformerEncoderLayer

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.1):
        super(LinearClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(self.input_size, self.num_classes)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_feature):
        output_feature = self.softmax(self.linear(self.dropout(input_feature)))
        return output_feature

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerEncoderClassifier, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            # x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class MLP_CLF(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_CLF, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc3 = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        tmp = self.fc2(self.relu(self.fc1(x)))
        out = self.softmax(self.fc3(tmp))
        return out

class Self_Gating_LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.1):
        super(Self_Gating_LinearClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.linear = nn.Linear(self.input_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_feature):
        x = self.linear(self.dropout(input_feature))
        self_gating_x = x * self.sigmoid(x)
        output_feature = self.softmax(self_gating_x)
        return output_feature
