import json
import csv
import os
import torch
import random
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, random_split, Subset
from dataset import *
from torch.nn.utils.rnn import pad_sequence
from tool.checkpoint import save_split_indices, load_split_indices

try:
    from tool.memory_utils import get_dataloader_config
    HAS_MEMORY_UTILS = True
except ImportError:
    HAS_MEMORY_UTILS = False

np.random.seed(666)

# 全局 DataLoader 配置（首次调用时初始化）
_DATALOADER_CONFIG = None

def get_global_dataloader_config():
    """获取全局 DataLoader 配置"""
    global _DATALOADER_CONFIG
    if _DATALOADER_CONFIG is None:
        if HAS_MEMORY_UTILS:
            _DATALOADER_CONFIG = get_dataloader_config()
        else:
            _DATALOADER_CONFIG = {
                'pin_memory': torch.cuda.is_available(),
                'num_workers': min(os.cpu_count() or 1, 4),
                'persistent_workers': False
            }
            print(f"[DataLoader Config] Using fallback settings: pin_memory={_DATALOADER_CONFIG['pin_memory']}, num_workers={_DATALOADER_CONFIG['num_workers']}")
    return _DATALOADER_CONFIG


def moji_collate_fn(batch):
    texts = [item['text'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    sa_labels = [item['sa'] for item in batch]

    # 对 input_ids 和 attention_mask 进行填充
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # 将 labels 和 sa_labels 堆叠成张量
    labels = torch.stack(labels)
    sa_labels = torch.stack(sa_labels)

    return {
        'text': texts,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'sa': sa_labels
    }


def calculate_dataset_distribution(dataset, corpus_type, client_datasets=[]):
    dataset_ratios = {}
    dataset_sizes = {}  # A new dictionary to store the actual sizes

    if corpus_type == "test":
        dataset_ratios = {}
        dataset_sizes = len(dataset)
        for data_point in dataset:
            dataset_name = data_point["tag"]
            if dataset_name not in dataset_ratios:
                dataset_ratios[dataset_name] = 0
            dataset_ratios[dataset_name] += 1
        for dataset_name in dataset_ratios:
            dataset_ratios[dataset_name] /= dataset_sizes

        logger.info(f"Testing Dataset Ratios: {dataset_ratios}; Sizes: {dataset_sizes}")

    else:
        for i, client_dataset in enumerate(client_datasets):
            total_data_points = len(client_dataset)
            dataset_ratios[i] = {}
            dataset_sizes[i] = {}  # Initialize for this client

            for data_point_index in client_dataset.indices:
                dataset_name = dataset[data_point_index]["tag"]
                dataset_ratios[i][dataset_name] = dataset_ratios[i].get(dataset_name, 0) + 1
                dataset_sizes[i][dataset_name] = dataset_sizes[i].get(dataset_name, 0) + 1  # Counting the data points

            for dataset_name in dataset_ratios[i]:
                dataset_ratios[i][dataset_name] /= total_data_points  # Getting the ratio

        # Printing the dataset ratios and actual sizes
        if corpus_type == "train":
            logger.info("Test dataset Ratios: %s", dataset_ratios)
            logger.info("Test dataset Sizes: %s", dataset_sizes)

    # batch_idxs_example = [
    #     [1, 2, 3, 4, 5],  # data for client 0
    #     [6, 7, 8, 9, 10],  # data for client 1
    #     # ... (other clients)
    # ]


# 我要保存的batch_idxs数据太长了，不能保存在excel的不同sheet中。因此我想将其保存到csv文件中，请你据此修改我的代码重写save_to_csv(batch_idxs, num_clients)，load_from_csv(num_clients)
def save_to_csv(batch_idxs, num_clients):
    # Define the filename based on the number of clients
    file_name = f"./csv/batch_idxs_num_clients_{num_clients}.csv"

    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    # Saving data to a CSV file
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for client_data in batch_idxs:
            writer.writerow(client_data)

    print(f"Data successfully saved to {file_name}")


def save_to_excel(batch_idxs, num_clients):
    # batch_idxs_example = [
    #     [1, 2, 3, 4, 5],  # data for client 0
    #     [6, 7, 8, 9, 10],  # data for client 1
    #     # ... (other clients)
    # ]
    file_name = "./json/batch_idxs_and_num_clients."
    sheet_name = f"{num_clients}"  # 固定的工作表名称，如果需要，每次调用可以更改此处来添加新的工作表

    # Convert batch_idxs list of lists into a DataFrame
    df = pd.DataFrame(batch_idxs)

    # 如果文件不存在，创建一个新的Excel文件
    if not os.path.isfile(file_name):
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:  # 使用默认的写入模式，即'w'
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # 如果文件已经存在，则以追加模式打开文件并添加新工作表
        with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:  # 这里改变为追加模式
            # 为了避免重复的sheet名称，您可能需要动态地设置sheet名称
            # 这里为了简化示例，我假设每个新数据都需要一个新的sheet
            df.to_excel(writer, index=False, sheet_name=sheet_name)


def load_from_excel(num_clients):
    file_name = "./json/batch_idxs_and_num_clients.xlsx"
    sheet_name = f"{num_clients}"  # 使用num_clients来确定sheet名称

    # 检查文件是否存在
    if not os.path.isfile(file_name):
        print(f"文件{file_name}不存在。")
        return None  # 或者根据您的需要处理这个情况，比如引发异常

    try:
        # 从指定的工作表中读取数据
        with pd.ExcelFile(file_name) as xls:
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
            else:
                print(f"'{sheet_name}'工作表不存在于{file_name}中。")
                return None  # 或者根据您的需要处理这个情况，比如引发异常

        # 如果需要，可以在此进行其他数据处理，比如验证数据完整性等

        # 将DataFrame转换为列表的列表，方便后续处理
        batch_idxs = df.values.tolist()
        return batch_idxs

    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        # 根据您的错误处理策略，您可以选择引发异常或返回None
        return None


def load_from_csv(num_clients):
    # Define the filename based on the number of clients
    file_name = f"./csv/batch_idxs_num_clients_{num_clients}.csv"

    # Check if the file exists
    if not os.path.isfile(file_name):
        print(f"File {file_name} does not exist.")
        return None

    # Reading data back from the CSV file
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)

        # Reconstructing the data
        batch_idxs = []
        for row in reader:
            # Since CSV stores everything as strings, we need to convert data back to integers
            int_row = [int(item) for item in row]
            batch_idxs.append(int_row)

    return batch_idxs


def distribute_by_dataset(dataset, cumulative_sizes, dataset_tag_beta_map, num_clients):
    # A function to distribute indices by dataset type based on cumulative_sizes
    idx_ranges = [(0, cumulative_sizes[0])]
    for i in range(1, len(cumulative_sizes)):
        idx_ranges.append((cumulative_sizes[i - 1], cumulative_sizes[i]))

    distributed_idxs = {tag: list(range(start, end)) for (start, end), tag in
                        zip(idx_ranges, dataset_tag_beta_map.keys())}
    return distributed_idxs


def split_data_for_clients(dataset, num_clients, dataset_tag_beta_map, cumulative_sizes):
    distributed_idxs = distribute_by_dataset(dataset, cumulative_sizes, dataset_tag_beta_map, num_clients)

    # Initialize batch_idxs as a list of empty lists for each client
    batch_idxs = [[] for _ in range(num_clients)]

    for tag, indices in distributed_idxs.items():
        beta = dataset_tag_beta_map[tag]  # Get the beta for this dataset type

        min_size = 0
        while min_size < 1:  # Ensure at least one sample per client
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = proportions / proportions.sum()  # Normalize
            min_size = np.min(proportions * len(indices))

        # Split indices among clients based on calculated proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_idxs = np.split(indices, proportions)

        # Extend each client's list of indices with the new indices
        for client_idx_list, new_indices in zip(batch_idxs, split_idxs):
            client_idx_list.extend(new_indices.tolist())  # Convert numpy array to list before extending

    return batch_idxs


def get_FL_dataloader(param_dict, dataset, num_clients, split_strategy="Uniform",
                        do_train=True, batch_size=64,
                        do_shuffle=True, num_workers=None, corpus_type="test"):

    # 获取智能 DataLoader 配置
    dl_config = get_global_dataloader_config()
    
    # 使用智能配置，允许用户通过参数覆盖
    effective_num_workers = num_workers if num_workers is not None else dl_config['num_workers']
    effective_pin_memory = dl_config['pin_memory']

    # algorithm = param_dict['algorithm']
    # if algorithm == "Centralized":
    #     partition_size = len(dataset) // num_clients
    #     lengths = [partition_size] * num_clients
    #     client_datasets = random_split(dataset, lengths, torch.Generator().manual_seed(666))
    #     trainloaders = []
    #     for ds in client_datasets:
    #         # trainloaders.append(
    #         #     DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=effective_num_workers, pin_memory=effective_pin_memory,
    #         #                collate_fn=moji_collate_fn))
    #         trainloaders.append(
    #             DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=effective_num_workers, pin_memory=effective_pin_memory,
    #                        ))
    #     return trainloaders, client_datasets

    # 尝试加载已保存的分割索引
    loaded_split_indices = load_split_indices(param_dict) if do_train else None

    if "Dirichlet" in split_strategy:

        beta = 0.5
        if split_strategy == "Dirichlet01":
            beta = 0.1
        elif split_strategy == "Dirichlet05":
            beta = 0.5
        elif split_strategy == "Dirichlet1":
            beta = 1
        elif split_strategy == "Dirichlet8":
            beta = 8

        if loaded_split_indices is not None:
            print("Loading saved split indices...")
            batch_idxs = [np.array(loaded_split_indices[i]) for i in range(num_clients)]
        else:
            print("Try to sperate the dataset...")
            if "Tabular_CLF" in param_dict["task"]:
                idxs = np.random.permutation(len(dataset))
                min_size = 0
                try_time = 0
                while min_size < 1:  # 每个客户至少拥有1个数据样本
                    if try_time <= 1000:
                        proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                        proportions = proportions / proportions.sum()
                        min_size = np.min(proportions * len(idxs))
                        try_time += 1
                        print(f"The {try_time}-th time separate the dataset, the min_size is : {min_size}")
                    else:
                        min_size = 1
            else:
                idxs = np.random.permutation(len(dataset))
                min_size = 0
                while min_size < 1:  # 每个客户至少拥有1个数据样本
                    proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                    proportions = proportions / proportions.sum()
                    min_size = np.min(proportions * len(idxs))

            print("Separating the dataset finish!!!")

            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            batch_idxs = np.split(idxs, proportions)

            if "Tabular_CLF" in param_dict["task"]:
                # 检查是否出现了没有任何数据的客户端，如果有，从数据量最多的客户端的里取一条数据作为弥补
                len_list = [len(item) for item in batch_idxs]  # 检查现在的数据集分布情况（各个客户端的数据量）
                empty_client_index_list = [index for index,item in enumerate(len_list) if item == 0]  # 找到需要填充的客户列表
                empty_client_count = len(empty_client_index_list)  # 计算需要填充的客户数目
                max_len_client_index = len_list.index(max(len_list))  # 找到最多数据量的客户
                max_len_client_batch_idx = batch_idxs[max_len_client_index]  # 得到最多数据量的客户的数据索引
                max_len_client_batch_idx_list = max_len_client_batch_idx.tolist()
                jackpot = random.sample(max_len_client_batch_idx_list, empty_client_count)  # 抽取需要用来“填空”的数据
                jackpot_index_list = [max_len_client_batch_idx_list.index(jack) for jack in jackpot]
                for i, item in enumerate(empty_client_index_list):  # 填空
                    batch_idxs[item] = np.append(batch_idxs[item], jackpot[i])
                batch_idxs[max_len_client_index] = np.delete(batch_idxs[max_len_client_index], [jackpot_index_list])  # 把填空过后的数据，从原始位置删除

            # 保存分割索引
            if do_train:
                split_indices = {i: batch_idxs[i] for i in range(num_clients)}
                save_split_indices(param_dict, split_indices)

        if do_train:
            client_datasets = [Subset(dataset, indices=batch_idxs[i]) for i in range(num_clients)]
            # trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle,
            #                            num_workers=effective_num_workers, pin_memory=effective_pin_memory, collate_fn=moji_collate_fn) for ds in client_datasets]
            trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle,
                                       num_workers=effective_num_workers, pin_memory=effective_pin_memory) for ds in client_datasets]
            return trainloaders, client_datasets

        else:
            calculate_dataset_distribution(dataset, corpus_type)

            # testloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle,
            #                         num_workers=effective_num_workers, pin_memory=effective_pin_memory, collate_fn=moji_collate_fn)
            testloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle,
                                    num_workers=effective_num_workers, pin_memory=effective_pin_memory)
            return testloader
    elif split_strategy == "Uniform":
        # Split training set into serval partitions to simulate the individual dataset
        if loaded_split_indices is not None:
            print("Loading saved split indices...")
            batch_idxs = [np.array(loaded_split_indices[i]) for i in range(num_clients)]
            if do_train:
                client_datasets = [Subset(dataset, indices=batch_idxs[i]) for i in range(num_clients)]
                trainloaders = [DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle,
                                           num_workers=effective_num_workers, pin_memory=effective_pin_memory) for ds in client_datasets]
                return trainloaders, client_datasets
        else:
            partition_size = len(dataset) // num_clients
            lengths = [partition_size] * num_clients

            remainder = len(dataset) - (partition_size * num_clients)
            lengths[-1] += remainder

            if do_train:
                client_datasets = random_split(dataset, lengths, torch.Generator().manual_seed(666))
                
                # 提取分割索引并保存
                batch_idxs = []
                for ds in client_datasets:
                    batch_idxs.append(np.array(ds.indices))
                
                split_indices = {i: batch_idxs[i] for i in range(num_clients)}
                save_split_indices(param_dict, split_indices)
                
                trainloaders = []
                for ds in client_datasets:
                    # trainloaders.append(
                    #     DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=effective_num_workers, pin_memory=effective_pin_memory,
                    #                collate_fn=moji_collate_fn))
                    trainloaders.append(
                        DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=effective_num_workers, pin_memory=effective_pin_memory,
                                   ))
                return trainloaders, client_datasets


            else:
                # calculate_dataset_distribution(dataset, corpus_type)
                # testloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle, num_workers=effective_num_workers, pin_memory=effective_pin_memory,
                #                         collate_fn=moji_collate_fn)
                testloader = DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle, num_workers=effective_num_workers, pin_memory=effective_pin_memory,
                                        )

                return testloader
    else:
        pass
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


