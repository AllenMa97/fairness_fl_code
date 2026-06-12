# https://arxiv.org/pdf/2109.05662
import os
import gc
import copy
import time
import torch
import pickle
import numpy as np
from transformers import BertTokenizer
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.client_selection import client_selection
from moudle.dataset import (MoJiDataset, BiosDataset, MTCDataset, CelebaDataset,
                            get_UTKFace_dataset, get_FairFace_dataset, get_LFWAPlus_dataset,
                            get_ADULT_dataset, get_COMPAS_dataset, get_DRUG_dataset, get_DUTCH_dataset)
from torch.utils.data import DataLoader, Subset
from tool.utils import FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value



def D_hat_θ(param_dict, client_dataset, client_model, device, tokenizer):
    # According to Eq 7., the D_hat_θ = L_hat_a,c - L_hat_b,c , where a & b are the value of the sensitive attribute.
    if "SENT_CLF" in param_dict["task"]:
        client_X = client_dataset.dataset.texts
    elif "IMG_CLF" in param_dict["task"]:
        client_X = client_dataset.dataset.img_names
    elif "Tabular_CLF" in param_dict["task"]:
        client_X = client_dataset.dataset.X

    client_y = client_dataset.dataset.labels
    try:
        client_s = client_dataset.dataset.protected
    except Exception:
        client_s = client_dataset.dataset.s1

    a, b = 1, 0

    c0, c1 = (np.array(client_y) == 0), (np.array(client_y) == 1)
    sa, sb = (np.array(client_s) == a), (np.array(client_s) == b)

    sa_c0, sa_c1 = sa * c0, sa * c1
    sb_c0, sb_c1 = sb * c0, sb * c1

    m_sa_c0, m_sa_c1 = sum(sa_c0), sum(sa_c1)
    m_sb_c0, m_sb_c1 = sum(sb_c0), sum(sb_c1)

    X_sa_c0, X_sa_c1, y_sa_c0, y_sa_c1 = [], [], [], []

    X_sb_c0, X_sb_c1, y_sb_c0, y_sb_c1 = [], [], [], []

    # 构建对应的数据集子集
    for index in range(len(client_y)):
        flag_1 = sa_c0[index]
        flag_2 = sa_c1[index]
        flag_3 = sb_c0[index]
        flag_4 = sb_c1[index]

        if flag_1:
            if "SENT_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                X_sa_c0.append(client_X[index])
            elif "IMG_CLF" in param_dict["task"]:
                X_sa_c0.append(index)
            y_sa_c0.append(client_y[index])
        if flag_2:
            if "SENT_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                X_sa_c1.append(client_X[index])
            elif "IMG_CLF" in param_dict["task"]:
                X_sa_c1.append(index)
            y_sa_c1.append(client_y[index])
        if flag_3:
            if "SENT_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                X_sb_c0.append(client_X[index])
            elif "IMG_CLF" in param_dict["task"]:
                X_sb_c0.append(index)
            y_sb_c0.append(client_y[index])
        if flag_4:
            if "SENT_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
                X_sb_c1.append(client_X[index])
            elif "IMG_CLF" in param_dict["task"]:
                X_sb_c1.append(index)
            y_sb_c1.append(client_y[index])

    # 四种数据分别计算损失
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    dataset_name = param_dict['dataset_name'].lower()
    if "moji".lower() in dataset_name:
        Dataset_func = MoJiDataset
    elif "bios".lower() in dataset_name:
        Dataset_func = BiosDataset
    elif "mtc".lower() in dataset_name:
        Dataset_func = MTCDataset
    elif "celeba".lower() in dataset_name:
        Dataset_func = CelebaDataset

    if "IMG_CLF" in param_dict["task"]:
        if "CelebA".lower() == param_dict["dataset"].lower():
            full_IMG_dataset = Dataset_func(
                data_dir=r'dataset/celeba',
                split='test'
            )
        elif "UTKFace".lower() == param_dict["dataset"].lower():
            _, full_IMG_dataset = get_UTKFace_dataset()

        elif "FairFace".lower() == param_dict["dataset"].lower():
            _, full_IMG_dataset = get_FairFace_dataset()

        elif "LFWA+".lower() == param_dict["dataset"].lower():
            _, full_IMG_dataset = get_LFWAPlus_dataset()

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
            _, _, _, testing_dataset = get_dataset(
                data_path,
                mask_s1_flag,
                mask_s2_flag,
                mask_s1_s2_flag)
        else:
            with open(pickle_path, 'rb') as r:
                pickle_dict = pickle.load(r)
                r.close()
            testing_dataset = pickle_dict['testing_dataset']


    # 第一种数据,s=a, c=0
    # logger.info("第一种数据,s=a, c=0")
    L_hat_ac0 = 0
    if len(X_sa_c0) != 0:
        with torch.no_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sa_c0,
                    labels=y_sa_c0,
                    protected=[a for _ in X_sa_c0],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif "IMG_CLF" in param_dict["task"]:
                testing_dataset = full_IMG_dataset
            testloader = DataLoader(testing_dataset, batch_size=param_dict['batch_size'], shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)
                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_ac0 += torch.sum(batch_loss) / m_sa_c0

    # 第二种数据,s=a, c=1
    # logger.info("第二种数据,s=a, c=1")
    L_hat_ac1 = 0
    if len(X_sa_c1) != 0:
        with torch.no_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sa_c1,
                    labels=y_sa_c1,
                    protected=[a for _ in X_sa_c1],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif "IMG_CLF" in param_dict["task"]:
                testing_dataset = full_IMG_dataset
            testloader = DataLoader(testing_dataset, batch_size=param_dict['batch_size'], shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)

                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_ac1 += torch.sum(batch_loss) / m_sa_c1

    # 第三种数据,s=b, c=0
    # logger.info("第三种数据,s=b, c=0")
    L_hat_bc0 = 0
    if len(X_sb_c0) != 0:
        with torch.no_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sb_c0,
                    labels=y_sb_c0,
                    protected=[a for _ in X_sb_c0],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif "IMG_CLF" in param_dict["task"]:
                testing_dataset = full_IMG_dataset
            testloader = DataLoader(testing_dataset, batch_size=param_dict['batch_size'], shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)
                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_bc0 += torch.sum(batch_loss) / m_sb_c0


    # 第四种数据,s=b, c=1
    # logger.info("第四种数据,s=b, c=1")
    L_hat_bc1 = 0
    if len(X_sb_c1) != 0:
        with torch.no_grad():
            if "SENT_CLF" in param_dict["task"]:
                testing_dataset = Dataset_func(
                    texts=X_sb_c1,
                    labels=y_sb_c1,
                    protected=[a for _ in X_sb_c1],
                    tokenizer=tokenizer,
                    max_len=param_dict["max_len"]
                )
            elif "IMG_CLF" in param_dict["task"]:
                testing_dataset = full_IMG_dataset
            testloader = DataLoader(testing_dataset, batch_size=param_dict['batch_size'], shuffle=True)
            for batch in testloader:
                # labels尺寸 [batch_size]
                labels = batch["labels"].to(device)
                if "SENT_CLF" in param_dict["task"]:
                    # input_ids尺寸 [batch_size, max_len]
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    # features尺寸 [batch_size, emb_dim]
                    # logits尺寸 [batch_size, category]
                    features, logits = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # activated_preds = logits.softmax(dim=1)
                    activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                    _, preds = torch.max(activated_preds, dim=1)
                    # batch_loss尺寸 [batch_size]
                    batch_loss = criterion(activated_preds, labels)

                elif "IMG_CLF" in param_dict["task"]:
                    imgs = batch["img"].to(device)
                    # preds尺寸 [batch_size, 1]
                    # features尺寸 [batch_size, emb_dim]
                    preds, features = client_model(imgs)
                    batch_loss = criterion(preds[:, 0], labels.float())

                elif "Tabular_CLF" in param_dict["task"]:
                    X = batch["X"].to(device)
                    # local_prediction尺寸 [batch_size, 1]
                    if "ANN" in str(type(client_model)):
                        local_prediction, features = client_model(X)
                    elif "LogisticRegression" in str(type(client_model)):
                        local_prediction = client_model(X)
                    else:
                        local_prediction = client_model(X)
                    batch_loss = criterion(local_prediction[:, 0], labels.float())

                L_hat_bc1 += torch.sum(batch_loss) / m_sb_c1

    del testing_dataset, testloader
    gc.collect()
    torch.cuda.empty_cache()

    L_hat_ac = L_hat_ac0 + L_hat_ac1
    L_hat_bc = L_hat_bc0 + L_hat_bc1

    return L_hat_ac + L_hat_bc, L_hat_ac - L_hat_bc


# FedFair直接训练的是全局模型

def FedFair(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]
    client_datasets_total_size = sum(client_datasets_size_list)


    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    optimizer = BERTCLF_Optimizer(
        method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
    optimizer.set_parameters(list(global_model.named_parameters()))

    total_gpu_seconds = 0

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024*1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # FedFair超参数
    β = 0.05
    γ = 0.001
    ϵ = 0.05
    λ_a, λ_b = 0.15, 1

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        idxs_users = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        client_loss_list = []
        client_D_hat_list = []

        # 记录GPU计算开始时间
        gpu_start_time = time.time()

        # Simulate Client Parallel
        for id in idxs_users:
            # Local Initialization
            # 下发模型
            logger.info("Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)


            # 修改后的Local Training
            for epoch in range(algorithm_epoch_T):
                # print("A")
                # print(torch.cuda.memory_summary())
                client_i_loss, D_hat_i_θ = D_hat_θ(param_dict, client_dataset_list[id], model, device, tokenizer)
                # print("B")
                # print(torch.cuda.memory_summary())
                # Equation 11
                first_term_in_Eq11 = (client_datasets_size_list[id] / client_datasets_total_size) * client_i_loss
                second_term_in_Eq11 = (λ_a - λ_b) * D_hat_i_θ / num_clients_K
                client_i_loss = first_term_in_Eq11 + second_term_in_Eq11
                # print("C")
                # print(torch.cuda.memory_summary())
                client_loss_list.append(client_i_loss)
                client_D_hat_list.append(D_hat_i_θ)
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {client_i_loss/ client_datasets_size_list[id]}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # 再走一个训练batch构建运算图
        tmp_loss = 0
        client_i_dataloader = training_dataloaders[idxs_users[0]]
        global_model.train()
        global_model.to(device)
        for batch_id, batch in enumerate(client_i_dataloader):
            # labels尺寸 [batch_size]
            labels = batch["labels"][:2].to(device)
            if "SENT_CLF" in param_dict["task"]:
                input_ids = batch["input_ids"][:2].to(device)
                attention_mask = batch["attention_mask"][:2].to(device)

                features, logits = global_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                activated_preds = logits
                _, preds = torch.max(activated_preds, dim=1)
                tmp_loss += criterion(activated_preds, labels)
                del input_ids, attention_mask, labels, features, logits, activated_preds, _, preds
            elif "IMG_CLF" in param_dict["task"]:
                imgs = batch["img"][:2].to(device)
                # preds尺寸 [batch_size, 1]
                # features尺寸 [batch_size, emb_dim]
                preds, features = global_model(imgs)
                tmp_loss += criterion(preds[:, 0], labels.float())
            elif "Tabular_CLF" in param_dict["task"]:
                X = batch["X"][:2].to(device)
                # local_prediction尺寸 [batch_size, 1]
                if "ANN" in str(type(global_model)):
                    local_prediction, features = global_model(X)
                elif "LogisticRegression" in str(type(global_model)):
                    local_prediction = global_model(X)
                else:
                    local_prediction = global_model(X)
                tmp_loss = criterion(local_prediction[:, 0], labels.float())

            gc.collect()
            torch.cuda.empty_cache()
            # 这里一定要break，因为仅仅只是为了构造运算图
            break

        # Parameter update by Equation 10
        global_loss = 0*tmp_loss.sum() + sum(client_loss_list)

        global_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录GPU计算结束时间
        gpu_end_time = time.time()

        common_term_of_the_first_term_in_Eq12_Eq13 = 1 - γ * β
        first_term_in_Eq12 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_a
        first_term_in_Eq13 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_b

        accumulation_of_client_D_hat_list = float(sum(client_D_hat_list))
        second_term_in_Eq12_Eq_13 = (β / len(idxs_users)) * accumulation_of_client_D_hat_list

        third_term_in_Eq12_Eq_13 = β * ϵ

        eq_12 = first_term_in_Eq12 + second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13
        eq_13 = first_term_in_Eq13 - second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13
        # 更新FedFair引入的参数
        λ_a = max(eq_12, 0)
        λ_b = max(eq_13, 0)

        # 没有到达最后一次通信轮次之前，都要做测试
        if (iter_t + 1) != param_dict['communication_round_I']:
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader,
                                                                   testing_dataset_len)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict,
                                                                                 testing_dataloader,
                                                                                 testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(
                    f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")

        total_gpu_seconds = gpu_end_time - gpu_start_time
        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = total_gpu_seconds / num_clients_K
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedFair.pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost
