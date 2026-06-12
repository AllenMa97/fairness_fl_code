import torch
import torch.nn as nn
# from pytorch_transformers import BertModel, BertConfig  # 旧版库
from transformers import BertModel

from tool.common import logger

# Google Original BERT
class OriBERT(nn.Module):
    def __init__(self, temp_dir, large_flag=False, cased_flag=False, finetune=False):
        super(OriBERT, self).__init__()

        # self.pretrained_model_name_or_path = 'bert-base-uncased'
        if (large_flag): # Large
            if (cased_flag):  # Cased
                self.pretrained_model_name_or_path = 'bert-large-cased' # 24-layer, 1024-hidden, 16-heads, 340M parameters
            else:
                self.pretrained_model_name_or_path = 'bert-large-uncased' # 24-layer, 1024-hidden, 16-heads, 340M parameters
        else:
            if (cased_flag):  # Cased
                self.pretrained_model_name_or_path = 'bert-base-cased'
            else:
                self.pretrained_model_name_or_path = 'bert-base-uncased'

        try:
            # self.PLM = BertModel.from_pretrained(
            #         pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            #         cache_dir=temp_dir)
            self.PLM = BertModel.from_pretrained(self.pretrained_model_name_or_path,temp_dir)

        except EnvironmentError:  # file not found
            # self.PLM = BertModel.from_pretrained(
            #     pretrained_model_name_or_path=self.pretrained_model_name_or_path)
            self.PLM = BertModel.from_pretrained(self.pretrained_model_name_or_path)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if (self.finetune):
            top_vec, _ = self.PLM(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.PLM(x, segs, attention_mask=mask)
        return top_vec
        
        
#Chinese BERT
class CNBERT(nn.Module):
    def __init__(self, temp_dir='', series='RoBERTa', finetune=False):
        super(CNBERT, self).__init__()

        if series == "BERT":
            # BERT Series 默认HFL CN BERT WWM EXT
            # self.pretrained_model_name_or_path = 'bert-base-chinese'    # Google Chinese BERT 110M参数量
            self.pretrained_model_name_or_path = 'hfl/chinese-bert-wwm'    # HFL(哈工大讯飞) Chinese BERT WWM
            # self.pretrained_model_name_or_path = 'hfl/chinese-bert-wwm-ext'    # HFL(哈工大讯飞) Chinese BERT WWM EXT
        elif series == "RoBERTa":
            # RoBERTa Series 默认HFL CN RoBERTa WWM EXT
            self.pretrained_model_name_or_path = 'hfl/chinese-roberta-wwm-ext'    # HFL(哈工大讯飞) Chinese RoBERTa WWM EXT 102M参数量
            # self.pretrained_model_name_or_path = 'hfl/chinese-roberta-wwm-ext-large'    # HFL(哈工大讯飞) Chinese RoBERTa WWM EXT Large 325M参数量
        elif series == "Tiny":
            # Tiny Series 默认HFL CN minirbt-h288
            # self.pretrained_model_name_or_path = 'ckiplab/albert-tiny-chinese'    # CKIP(台湾中研院) 繁体 ALBERT
            # self.pretrained_model_name_or_path = 'hfl/rbt4-h312'    # HFL(哈工大讯飞) rbt4-h312 与TinyBERT同大小，共4层312隐层尺寸12头，11.4M参数量
            # self.pretrained_model_name_or_path = 'hfl/minirbt-h256'    # HFL(哈工大讯飞) MiniRBT-h256 共6层256隐层尺寸8头，10.4M参数量
            self.pretrained_model_name_or_path = 'hfl/minirbt-h288'    # HFL(哈工大讯飞) MiniRBT-h288 共6层288隐层尺寸8头，12.3M参数量
        else:
            self.pretrained_model_name_or_path = 'hfl/chinese-roberta-wwm-ext'    # HFL(哈工大讯飞) Chinese RoBERTa WWM EXT 102M参数量

        try:
            logger.info(f"Try to load the PLM from temp_dir : {temp_dir}")
            self.PLM = BertModel.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                cache_dir=temp_dir)

        except EnvironmentError:  # file not found
            logger.info(f"Try to download the PLM: {self.pretrained_model_name_or_path}")
            self.PLM = BertModel.from_pretrained(self.pretrained_model_name_or_path)
        self.finetune = finetune


    def forward(self, input_ids, attention_mask):
        if (self.finetune):
            self.train()
            result = self.PLM(input_ids, attention_mask=attention_mask)
        else:
            self.eval()
            with torch.no_grad():
                result = self.PLM(input_ids, attention_mask=attention_mask)
        return result

if __name__ == '__main__':
    # print("下载PLM")
    # PLM = OriBERT("../../save/PLM", large_flag=False, cased_flag=False, finetune=False)
    # PLM = OriBERT("../../save/PLM", large_flag=False, cased_flag=False, finetune=True)
    # PLM = OriBERT("../../save/PLM", large_flag=False, cased_flag=True, finetune=True)
    # PLM = OriBERT("../../save/PLM", large_flag=True, cased_flag=True, finetune=True)
    # PLM = OriBERT("../../save/PLM", large_flag=True, cased_flag=False, finetune=False)
    # PLM = OriBERT("../../save/PLM", large_flag=True, cased_flag=False, finetune=True)


    # PLM = CNBERT("../../save/PLM", series="BERT", finetune=False)
    PLM = CNBERT("../save/PLM", series="RoBERTa", finetune=False)
    # PLM = CNBERT("../../save/PLM", series="Tiny", finetune=False)



    print("Finish!")