import torch.nn as nn
from torch.nn.init import xavier_uniform_

from hypothesis.backup.Classifier import LinearClassifier, MLP_CLF, Self_Gating_LinearClassifier
from hypothesis.backup.TextCNN import TextCNN


class BERT_Linear_TextCNN(nn.Module):
    def __init__(self, BERT, num_classes, dropout_rate):
        super(BERT_Linear_TextCNN, self).__init__()
        self.BERT = BERT
        self.input_size = self.BERT.PLM.config.hidden_size

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.frozen_ext_layer = LinearClassifier(input_size=self.input_size, num_classes=self.num_classes, dropout_rate=self.dropout_rate)
        self.frozen_MLP = MLP_CLF(input_size=self.input_size, num_classes=self.num_classes, hidden_size=int(self.input_size/2))
        self.frozen_sg_ext_layer = Self_Gating_LinearClassifier(input_size=self.input_size, num_classes=self.num_classes, dropout_rate=self.dropout_rate)
        for p in self.frozen_ext_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            p.requires_grad = False

        for p in self.frozen_MLP.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            p.requires_grad = False

        self.Text_CNN_with_linear_layer = TextCNN(num_classes=self.num_classes, input_size=self.input_size)
        for p in self.Text_CNN_with_linear_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            p.requires_grad = True

    def forward(self, PLM_input_dict, head_style="frozen_ext_layer"):
        try:
            # 适用于BertTokenizer输出的结果
            # [batch_size, inner_batch_token_length, PLM_hidden_size]
            embs = self.BERT(**PLM_input_dict).last_hidden_state
        except Exception:
            # 适用于自封装的结果
            # [batch_size, inner_batch_token_length, PLM_hidden_size]
            embs = self.BERT(PLM_input_dict['input_ids'],PLM_input_dict['attention_mask']).last_hidden_state

        if 'frozen_ext_layer' in head_style:
            seqence_output  = self.frozen_ext_layer(embs[:,0,])  # [batch_size, num_classes]
        elif 'TextCNN' in head_style:
            seqence_output = self.Text_CNN_with_linear_layer(embs)  # [batch_size, inner_batch_token_length, PLM_hidden_size]
        elif 'MLP_CLF' in head_style:
            seqence_output  = self.frozen_MLP(embs[:,0,])  # [batch_size, num_classes]

        else:
            seqence_output  = self.Text_CNN_with_linear_layer(embs)  # [batch_size, inner_batch_token_length, PLM_hidden_size]
        return embs, seqence_output


class BERT_Linear_MLP(nn.Module):
    def __init__(self, BERT, num_classes, dropout_rate):
        super(BERT_Linear_MLP, self).__init__()
        self.BERT = BERT
        self.input_size = self.BERT.PLM.config.hidden_size

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.CLF = LinearClassifier(input_size=self.input_size, num_classes=self.num_classes, dropout_rate=self.dropout_rate)
        self.MLP = MLP_CLF(input_size=self.input_size, num_classes=self.num_classes, hidden_size=int(self.input_size/2))
        self.SG = Self_Gating_LinearClassifier(input_size=self.input_size, num_classes=self.num_classes, dropout_rate=self.dropout_rate)
        for p in self.CLF.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            # p.requires_grad = False

        for p in self.MLP.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            # p.requires_grad = False

        for p in self.SG.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, PLM_input_dict, head_style="MLP"):
        try:
            # 适用于BertTokenizer输出的结果
            # [batch_size, inner_batch_token_length, PLM_hidden_size]
            embs = self.BERT(**PLM_input_dict).last_hidden_state
        except Exception:
            # 适用于自封装的结果
            # [batch_size, inner_batch_token_length, PLM_hidden_size]
            embs = self.BERT(PLM_input_dict['input_ids'],PLM_input_dict['attention_mask']).last_hidden_state
        if 'CLF' in head_style:
            seqence_output  = self.CLF(embs[:,0,])  # [batch_size, num_classes]
        elif 'MLP' in head_style:
            seqence_output  = self.MLP(embs[:,0,])  # [batch_size, num_classes]
        elif 'SG' in head_style:
            seqence_output  = self.SG(embs[:,0,])  # [batch_size, num_classes]
        else:
            seqence_output  = self.CLF(embs[:,0,])  # [batch_size, num_classes]
        return embs, seqence_output
