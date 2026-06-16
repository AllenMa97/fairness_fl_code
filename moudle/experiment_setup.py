import os
import sys
import torch
import pickle
import pandas as pd
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from hypothesis.BERTCLASSIFIER import BertClassifier
from hypothesis.CNNCLASSIFIER import RegularCNN
from hypothesis.ANNCLASSIFIER import RegularANN

from hypothesis.LogisticRegression import RegularLogisticRegression
from moudle.dataloader import get_FL_dataloader
from moudle.dataset import MoJiDataset, BiosDataset, MTCDataset, CelebaDataset
from moudle.dataset import get_UTKFace_dataset, get_ADULT_dataset, get_COMPAS_dataset, get_DRUG_dataset, get_DUTCH_dataset, get_FairFace_dataset, get_LFWAPlus_dataset

from tool.logger import *

# 数据集路径检查表：数据集名 -> 需要存在的关键路径
_DATASET_REQUIRED_PATHS = {
    "celeba": ["dataset/celeba/Img/img_align_celeba/"],
    "utkface": ["dataset/UTKFace/img/"],
    "fairface": ["dataset/FairFace/train/", "dataset/FairFace/fairface_label_train.csv"],
    "lfwaplus": ["dataset/LFWA+/lfw/"],
    "adult": ["dataset/ADULT/adult.data"],
    "compas": ["dataset/COMPAS/compas-scores-two-years.csv"],
    "drug": ["dataset/DRUG/drug_consumption.data"],
    "dutch": ["dataset/DUTCH/dutch_census_2001.arff"],
    "bios": ["dataset/bios/train.parquet"],
    "moji": ["dataset/moji/train.parquet"],
}


def _check_dataset_exists(dataset_name):
    """检查数据集是否存在，不存在则自动调用 setup_data.py 下载"""
    key = dataset_name.lower()
    if key not in _DATASET_REQUIRED_PATHS:
        return True
    missing = [p for p in _DATASET_REQUIRED_PATHS[key] if not os.path.exists(p)]
    if not missing:
        return True

    print(f"\n[SETUP] 数据集 '{dataset_name}' 缺少必要文件，正在自动下载...")
    import subprocess
    setup_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup_data.py")
    result = subprocess.run([sys.executable, setup_script, "--datasets", dataset_name], capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] 数据集 '{dataset_name}' 自动下载失败，请手动运行: python setup_data.py --datasets {dataset_name}")
        return False

    # 下载后再检查一次
    missing = [p for p in _DATASET_REQUIRED_PATHS[key] if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] 数据集 '{dataset_name}' 下载后仍缺少文件: {missing}")
        return False
    print(f"[SETUP] 数据集 '{dataset_name}' 下载完成!\n")
    return True


def Experiment_Create_dataset(param_dict):
    dataset_name = param_dict['dataset_name'].lower()

    if not _check_dataset_exists(dataset_name):
        raise FileNotFoundError(f"数据集 '{dataset_name}' 不存在，请先运行 python setup_data.py --datasets {dataset_name}")

    # tail_index = None
    # tail_index = 2000
    try:
        if 'system_data_count' in list(param_dict.keys()):
            tail_index = int(param_dict['system_data_count'])
            logger.info(f"tail_index is : {tail_index}")
    except:
        tail_index = None


    if "SENT_CLF" in param_dict["task"]:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if "moji".lower() in dataset_name:
            df_train = pd.read_parquet(r'dataset/moji/train.parquet')[:tail_index]
            df_test = pd.read_parquet(r'dataset/moji/test.parquet')[:tail_index]

            # 编码标签
            le = LabelEncoder()
            df_train['encoded_label'] = le.fit_transform(df_train['label'])
            df_test['encoded_label'] = le.transform(df_test['label'])
            train_texts = df_train['text']
            train_labels = df_train['encoded_label']
            train_sa = df_train['sa']
            test_texts = df_test['text']
            test_labels = df_test['encoded_label']
            test_sa = df_test['sa']
            training_dataset = MoJiDataset(
                texts=train_texts.tolist(),
                labels=train_labels.tolist(),
                protected=train_sa.tolist(),
                tokenizer=tokenizer,
                max_len=param_dict["max_len"],
                cache_name="moji", cache_split="train"
            )

            testing_dataset = MoJiDataset(
                texts=test_texts.tolist(),
                labels=test_labels.tolist(),
                protected=test_sa.tolist(),
                tokenizer=tokenizer,
                max_len=param_dict["max_len"],
                cache_name="moji", cache_split="test"
            )
            validation_dataset = None
            # param_dict["le_class"] = len(le.classes_)
            param_dict["le_class"] = 2

        # elif "MTC".lower() in dataset_name:
        #     df_train = pd.read_csv(r'dataset/MTC/English/train.tsv', sep='\t', na_values='x')[:1000]
        #     df_test = pd.read_csv(r'dataset/MTC/English/test.tsv', sep='\t', na_values='x')[:1000]
        #
        #     # 根据原论文Github操作，手动加特殊字符
        #     df_train.text = df_train.text.apply(lambda x: tokenizer.decode(x))
        #
        #     df_train.text = df_train.text.apply(lambda x: '[CLS] ' + x + ' [SEP]')
        #     df_test.text = df_test.text.apply(lambda x: '[CLS] ' + x + ' [SEP]')
        #     df_train.text = df_train.text.apply(lambda x: tokenizer.tokenize(x))
        #     df_test.text = df_test.text.apply(lambda x: tokenizer.tokenize(x))
        #
        #
        #     # 根据原论文Github操作，手动生成索引并填充 convert to indices and pad the sequences
        #     max_len = 25 # 原论文的Github代码给的是25，但是经过测试应该是有bug，文本开头出现大量0编码
        #     df_train.text = df_train.text.apply(lambda x: pad_sequences([tokenizer.convert_tokens_to_ids(x)], maxlen=max_len, dtype="long")[0])
        #     df_test.text = df_test.text.apply(lambda x: pad_sequences([tokenizer.convert_tokens_to_ids(x)], maxlen=max_len, dtype="long")[0])
        #
        #     # 根据原论文Github操作，手动生成注意力掩码 create attention masks
        #     train_attention_masks = []
        #     for seq in df_train.text:
        #         seq_mask = [float(idx > 0) for idx in seq]
        #         train_attention_masks.append(seq_mask)
        #     test_attention_masks = []
        #     for seq in df_test.text:
        #         seq_mask = [float(idx > 0) for idx in seq]
        #         test_attention_masks.append(seq_mask)
        #
        #
        #     # 编码标签
        #     le = LabelEncoder()
        #     df_train = df_train[df_train['gender'].isin(['0', '1'])]
        #     df_test = df_test[df_test['gender'].isin(['0', '1'])]
        #     df_train['encoded_label'] = le.fit_transform(df_train['label'])
        #     df_test['encoded_label'] = le.transform(df_test['label'])
        #
        #     train_texts = df_train['text'] # 注意，这个数据集的text是已经被编码过的，不是纯文本
        #
        #
        #     train_labels = df_train['encoded_label']
        #     # 具体以哪个属性作为受保护的属性，在这个数据集中是可以选择的，具体有年龄，国籍，性别，种族，详情参考https://arxiv.org/pdf/2002.10361
        #     train_protected = df_train['gender'].astype(int)
        #
        #     test_texts = df_test['text']
        #     test_labels = df_test['encoded_label']
        #     # 具体以哪个属性作为受保护的属性，在这个数据集中是可以选择的，具体有年龄，国籍，性别，种族，详情参考https://arxiv.org/pdf/2002.10361
        #     test_protected = df_test['gender'].astype(int)
        #
        #     # print(len(train_texts), len(train_labels), len(train_protected))
        #     training_dataset = MTCDataset(
        #         input_ids=train_texts.tolist(),
        #         masks=train_attention_masks,
        #         labels=train_labels.tolist(),
        #         protected=train_protected.tolist()
        #     )
        #
        #     testing_dataset = MTCDataset(
        #         input_ids=test_texts.tolist(),
        #         masks=test_attention_masks,
        #         labels=test_labels.tolist(),
        #         protected=test_protected.tolist()
        #     )
        #     validation_dataset = None
        #     param_dict["le_class"] = len(le.classes_)

        elif "bios".lower() in dataset_name:
            df_train = pd.read_parquet('dataset/bios/train.parquet')[:tail_index]
            df_test = pd.read_parquet('dataset/bios/test.parquet')[:tail_index]

            # BIOS数据集本来是28分类，根据huggingface dataset网站的统计，把4个比例较大的类别(2,18,19,21)归为一类（占54.50%），剩下的归为另一类，形成二分类任务
            # 参考https://huggingface.co/datasets/LabHC/bias_in_bios
            def bios_binary(x):
                if int(x) in [2, 18, 19, 21]:
                    return 1
                else:
                    return 0

            # 编码标签
            le = LabelEncoder()
            df_train['encoded_label'] = le.fit_transform(df_train['profession'])
            df_test['encoded_label'] = le.transform(df_test['profession'])

            train_texts = df_train['hard_text']
            train_labels = df_train['encoded_label'].apply(lambda x: bios_binary(x))
            train_protected = df_train['gender']
            test_texts = df_test['hard_text']
            test_labels = df_test['encoded_label'].apply(lambda x: bios_binary(x))
            test_protected = df_test['gender']
            training_dataset = BiosDataset(
                texts=train_texts.tolist(),
                labels=train_labels.tolist(),
                protected=train_protected.tolist(),
                tokenizer=tokenizer,
                max_len=param_dict["max_len"],
                cache_name="bios", cache_split="train"
            )

            testing_dataset = BiosDataset(
                texts=test_texts.tolist(),
                labels=test_labels.tolist(),
                protected=test_protected.tolist(),
                tokenizer=tokenizer,
                max_len=param_dict["max_len"],
                cache_name="bios", cache_split="test"
            )
            validation_dataset = None
            param_dict["le_class"] = 2
    elif "IMG_CLF" in param_dict["task"]:

        if "celeba".lower() in dataset_name:
            path = r'dataset/celeba'
            training_dataset = CelebaDataset(data_dir=path, split='train')
            testing_dataset = CelebaDataset(data_dir=path, split='test')
            validation_dataset = CelebaDataset(data_dir=path, split='val')

            param_dict["le_class"] = 2

        elif "UTKFace".lower() in dataset_name:
            training_dataset, testing_dataset = get_UTKFace_dataset()
            validation_dataset = None

            param_dict["le_class"] = 2

        elif "FairFace".lower() in dataset_name:
            training_dataset, testing_dataset = get_FairFace_dataset()
            validation_dataset = None

            param_dict["le_class"] = 2

        elif "LFWA+".lower() in dataset_name:
            training_dataset, testing_dataset = get_LFWAPlus_dataset()
            validation_dataset = None

            param_dict["le_class"] = 2

        # 从数据集采样获取图像尺寸，存入 param_dict（供 dataset distillation 等算法使用）
        sample = training_dataset[0]['img'] if isinstance(training_dataset[0], dict) else training_dataset[0][0]
        param_dict['img_channels'] = sample.shape[0]
        param_dict['img_height'] = sample.shape[1]
        param_dict['img_width'] = sample.shape[2]
        param_dict['img_shape'] = tuple(sample.shape)

    elif "Tabular_CLF" in param_dict["task"]:
        mask_s1_flag = False
        mask_s2_flag = False
        mask_s1_s2_flag = False
        if "ADULT".lower() in dataset_name:
            pickle_path = "./dataset/ADULT/ADULT.pickle"
            data_path = "./dataset/ADULT"
            get_dataset = get_ADULT_dataset
        elif "COMPAS".lower() in dataset_name:
            pickle_path = "./dataset/COMPAS/COMPAS.pickle"
            data_path = "./dataset/COMPAS"
            get_dataset = get_COMPAS_dataset
        elif "DRUG".lower() in dataset_name:
            pickle_path = "./dataset/DRUG/DRUG.pickle"
            data_path = "./dataset/DRUG"
            get_dataset = get_DRUG_dataset
        elif "DUTCH".lower() in dataset_name:
            pickle_path = "./dataset/DUTCH/DUTCH.pickle"
            data_path = "./dataset/DUTCH"
            get_dataset = get_DUTCH_dataset
        if not os.path.exists(pickle_path):
            training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset = get_dataset(data_path,
                                                                                                                  mask_s1_flag,
                                                                                                                  mask_s2_flag,
                                                                                                                  mask_s1_s2_flag)
            pickle_dict = {
                "training_dataset": training_dataset,
                "positive_training_dataset": positive_training_dataset,
                "negative_training_dataset": negative_training_dataset,
                "testing_dataset": testing_dataset,
            }
            with open(pickle_path, 'wb') as p:
                pickle.dump(pickle_dict, p)
                p.close()
        else:
            try:
                with open(pickle_path, 'rb') as r:
                    pickle_dict = pickle.load(r)
                    r.close()
                training_dataset = pickle_dict['training_dataset']
                positive_training_dataset = pickle_dict['positive_training_dataset']
                negative_training_dataset = pickle_dict['negative_training_dataset']
                testing_dataset = pickle_dict['testing_dataset']
            except Exception:
                training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset = get_dataset(
                    data_path,
                    mask_s1_flag,
                    mask_s2_flag,
                    mask_s1_s2_flag)
                pickle_dict = {
                    "training_dataset": training_dataset,
                    "positive_training_dataset": positive_training_dataset,
                    "negative_training_dataset": negative_training_dataset,
                    "testing_dataset": testing_dataset,
                }
                with open(pickle_path, 'wb') as p:
                    pickle.dump(pickle_dict, p)
                    p.close()
        nn_input_size = training_dataset.X.shape[1]
        param_dict['nn_input_size'] = nn_input_size

        validation_dataset = None

    return training_dataset, validation_dataset, testing_dataset



def Experiment_Create_dataloader(param_dict, training_dataset, validation_dataset, testing_dataset,
                                 split_strategy="Uniform"):
    num_clients_K = param_dict['num_clients_K']
    batch_size = param_dict['batch_size']
    test_batch_size = param_dict['test_batch_size']

    training_dataloaders, client_dataset_list = get_FL_dataloader(param_dict,
                                                                  training_dataset, num_clients_K,
                                                                  split_strategy=split_strategy,
                                                                  do_train=True, batch_size=batch_size,
                                                                  num_workers=0, do_shuffle=True,
                                                                  corpus_type="train"
                                                                  )

    testing_dataloader = get_FL_dataloader(param_dict,
                                           testing_dataset, num_clients_K, split_strategy="Uniform",
                                           do_train=False, batch_size=test_batch_size, num_workers=0,
                                           corpus_type="test"
                                           )
    # print(training_dataloaders)
    # print(testing_dataloader)

    # return training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader
    return training_dataloaders, client_dataset_list, testing_dataloader


def Experiment_Create_model(param_dict):
    logger.info("Model construction")
    # param_dict["le_class"]  Label Encoder's Class, setted by  Experiment_Create_Dataset

    if "SENT_CLF" in param_dict["task"]:
        model = BertClassifier(n_classes=param_dict["le_class"])
        param_dict['emb_dim'] = 768  # BERT hidden size
    elif "IMG_CLF" in param_dict["task"]:
        model = RegularCNN()
        param_dict['emb_dim'] = 512  # CNN最后一层特征维度
    elif "Tabular_CLF" in param_dict["task"]:
        if 'LogisticRegression' in param_dict["model_type"]:
            model = RegularLogisticRegression(input_size=param_dict['nn_input_size'])
        elif 'ANN' in param_dict["model_type"]:
            model = RegularANN(input_size=param_dict['nn_input_size'])
        else:
            model = RegularANN(input_size=param_dict['nn_input_size'])
        param_dict['emb_dim'] = param_dict['nn_input_size']  # 表格数据的特征维度，因数据集而异

    # torch.compile 可选加速（默认关闭，需 opt-in）
    # SENT_CLF (BERT): mode="reduce-overhead" 减少动态 shape 重编译
    # IMG_CLF (CNN):  默认 mode 即可
    # Tabular_CLF:    模型太小，compile 无收益，跳过
    if param_dict.get('use_compile', False):
        try:
            if hasattr(torch, 'compile'):
                compile_mode = "reduce-overhead" if "SENT_CLF" in param_dict["task"] else "default"
                model = torch.compile(model, mode=compile_mode)
                logger.info(f"torch.compile() enabled (mode={compile_mode})")
            else:
                logger.warning("torch.compile not available (requires PyTorch >= 2.0)")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}, falling back to eager mode")

    # model.to(param_dict['device'])
    return model


def Experiment_Reload_model(checkpoint_path):
    model = torch.load(checkpoint_path, weights_only=False)
    return model
