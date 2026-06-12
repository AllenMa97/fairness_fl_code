import torch
from transformers import BertModel

class BertClassifier(torch.nn.Module):
    def __init__(self, n_classes, pooled_output_flag=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(p=0.1)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.pooled_output_flag = pooled_output_flag

        # 为scaffold算法提供的控制变量
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

    def only_PLM_forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        clf_output = outputs[0][:, 0, :]
        pooled_output = outputs[1]

        if self.pooled_output_flag:
            feature = pooled_output
        else:
            feature = clf_output

        return feature

    def only_clf_forward(self, feature):
        dropped_feature = self.drop(feature)
        logit = self.out(dropped_feature)
        return feature, logit

    def latent_forward(self, inputs_embeds, attention_mask, token_type_ids):
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids = token_type_ids
        )
        clf_output = outputs[0][:, 0, :]
        pooled_output = outputs[1]

        if self.pooled_output_flag:
            feature = pooled_output
        else:
            feature = clf_output

        # del outputs, clf_output, pooled_output

        dropped_feature = self.drop(feature)
        logit = self.out(dropped_feature)
        return feature, logit

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        clf_output = outputs[0][:, 0, :]
        pooled_output = outputs[1]

        if self.pooled_output_flag:
            feature = pooled_output
        else:
            feature = clf_output

        dropped_feature = self.drop(feature)
        logit = self.out(dropped_feature)
        return feature, logit