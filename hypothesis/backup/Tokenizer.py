from transformers import BertTokenizer

def getCNTokenizer(series='RoBERTa'):

    if series == "BERT":
        pretrained_model_name_or_path = 'hfl/chinese-bert-wwm'  # HFL(哈工大讯飞) Chinese BERT WWM
    elif series == "RoBERTa":
        pretrained_model_name_or_path = 'hfl/chinese-roberta-wwm-ext'  # HFL(哈工大讯飞) Chinese RoBERTa WWM EXT 102M参数量
    elif series == "Tiny":
        pretrained_model_name_or_path = 'hfl/minirbt-h288'  # HFL(哈工大讯飞) MiniRBT-h288 共6层288隐层尺寸8头，12.3M参数量
    else:
        pretrained_model_name_or_path = 'hfl/chinese-roberta-wwm-ext'  # HFL(哈工大讯飞) Chinese RoBERTa WWM EXT 102M参数量

    CNTokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    return CNTokenizer
