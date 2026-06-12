import copy
import os
import gc
import time
import torch
import math
import numpy as np
from tool.logger import *
from algorithm.Optimizers import BERTCLF_Optimizer
from tool.utils import (get_parameters, set_parameters, communication_cost_simulated_by_beta_distribution, get_HM_by_two_value,
                        FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF)


def get_argmax_v(param_dict, idxs_users, client_model_path_list, mask_s1_flag, training_dataset, client_dataset_list,
                 r_hat_p0, r_hat_p1, device, γ_k_style):

    training_dataset_size = len(training_dataset)

    j_hat_0_0, j_hat_0_1, j_hat_1_0, j_hat_1_1 = 0, 0, 0, 0
    u_hat_0, u_hat_1 = 0, 0

    with torch.no_grad():
        for index, id in enumerate(idxs_users):
            selected_model = torch.load(client_model_path_list[index], weights_only=False).to(device)  # 持久化
            selected_model.eval()
            if "SENT_CLF" in param_dict["task"]:
                # TODO 这里还没有调过，不知道有没有bug
                # 有可能一次性推理多个句子，会爆显存，最稳妥还是逐个逐个推理然后再合并
                client_input_ids = torch.stack([training_dataset[idx]['input_ids'] for idx in client_dataset_list[id].indices]).to(device)
                client_attention_mask = torch.stack([training_dataset[idx]['attention_mask'] for idx in client_dataset_list[id].indices]).to(device)
                s = torch.tensor([training_dataset[idx]['protected'] for idx in client_dataset_list[id].indices])
                __, y_hat_θ = selected_model(
                    input_ids=client_input_ids,
                    attention_mask=client_attention_mask
                )
                # y_hat_θ尺寸 需要长成【样本数，布尔】的形式

            elif "IMG_CLF" in param_dict["task"]:
                s = torch.tensor([training_dataset[idx]['protected'] for idx in client_dataset_list[id].indices])
                # 一次性推理多张图片，可能爆显存
                # client_imgs = torch.stack([training_dataset[idx]['img'] for idx in client_dataset_list[id].indices]).to(device)
                # y_hat_θ = (selected_model(client_imgs)[0] >= 0.5).reshape(-1).to(device)
                # 逐个逐个推理然后再合并
                tmp_list = []
                for idx in client_dataset_list[id].indices:
                    tmp_pred = selected_model(training_dataset[idx]['img'].unsqueeze(0).to(device))[0]>=0.5
                    tmp_list.append(tmp_pred)
                y_hat_θ = torch.concatenate(tmp_list,dim=0).reshape(-1).to(device)

            elif "Tabular_CLF" in param_dict["task"]:
                # if mask_s1_flag:
                #     # Sensitive attribute 2
                #     s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[id].indices])
                # else:
                #     # Sensitive attribute 1
                #     s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[id].indices])

                s = torch.tensor([training_dataset[idx]['protected'] for idx in client_dataset_list[id].indices])

                client_X = torch.tensor(np.array([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices])).to(device)
                # local_prediction尺寸 [batch_size, 1]
                if "ANN" in str(type(selected_model)):
                    local_prediction, __ = selected_model(client_X)
                elif "LogisticRegression" in str(type(selected_model)):
                    local_prediction = selected_model(client_X)
                else:
                    local_prediction = selected_model(client_X)
                y_hat_θ = (local_prediction >= 0.5).reshape(-1).to(device)


            if "client" in γ_k_style: # uniform over client
                γ_k = float(1 / len(idxs_users))
            else: # uniform over distribution
                if "SENT_CLF" in param_dict["task"]:
                    γ_k = float(len(client_X) / training_dataset_size)
                elif "IMG_CLF" in param_dict["task"]:
                    γ_k = float(len(y_hat_θ) / training_dataset_size)
                elif "Tabular_CLF" in param_dict["task"]:
                    γ_k = float(len(y_hat_θ) / training_dataset_size)

            j_bar_0_0 = get_j_bar_c_p(y_hat_θ, 0, s, 0, device)
            j_bar_0_1 = get_j_bar_c_p(y_hat_θ, 0, s, 1, device)
            j_bar_1_0 = get_j_bar_c_p(y_hat_θ, 1, s, 0, device)
            j_bar_1_1 = get_j_bar_c_p(y_hat_θ, 1, s, 1, device)

            u_bar_0 = get_u_bar_c(y_hat_θ, 0, device)
            u_bar_1 = get_u_bar_c(y_hat_θ, 1, device)

            j_hat_0_0 += γ_k * j_bar_0_0
            j_hat_0_1 += γ_k * j_bar_0_1
            j_hat_1_0 += γ_k * j_bar_1_0
            j_hat_1_1 += γ_k * j_bar_1_1
            u_hat_0 += γ_k * u_bar_0
            u_hat_1 += γ_k * u_bar_1

        j_hat_c0_p0 = j_hat_0_0
        j_hat_c0_p1 = j_hat_0_1
        j_hat_c1_p0 = j_hat_1_0
        j_hat_c1_p1 = j_hat_1_1
        u_hat_c0 = u_hat_0
        u_hat_c1 = u_hat_1

        q_00 = get_q_c_p(j_hat_c0_p0, u_hat_c0, r_hat_p0, device)
        q_01 = get_q_c_p(j_hat_c0_p1, u_hat_c0, r_hat_p1, device)
        q_10 = get_q_c_p(j_hat_c1_p0, u_hat_c1, r_hat_p0, device)
        q_11 = get_q_c_p(j_hat_c1_p1, u_hat_c1, r_hat_p1, device)
        Q_hat = torch.tensor([
            [q_00, q_01],
            [q_10, q_11]
        ]).to(device)

        u, s, v = torch.linalg.svd(Q_hat)

        second_singular_vector_of_Q_hat = v[1].reshape(-1, 1).to(device)
    return second_singular_vector_of_Q_hat


def get_communication_idxs_list(num_clients_K, straggler_rate_α, descending_order_list):
    idxs_users = [i for i in range(num_clients_K)]
    straggler_ids = []
    if straggler_rate_α != 0:
        straggle_count = math.ceil(straggler_rate_α * num_clients_K)  # Round up the straggler count
        for tmp in range(straggle_count):
            straggler_id = descending_order_list[tmp]
            straggler_ids.append(straggler_id)
            # Remove stragglers
            idxs_users.remove(straggler_id)  # .remove(内容) or .pop(索引)
    return idxs_users, straggler_ids


def get_gamma_k_list(γ_k_style, client_datasets_size_list, num_clients_K):
    if "uniform_distribution" in γ_k_style:
        # uniform over distribution, γ_k =  n_k / n
        γ_denominator = sum(client_datasets_size_list)
    else:
        # uniform over client, γ_k = 1 / K
        γ_denominator = num_clients_K

    γ_k_list = []
    for i in range(num_clients_K):
        if "uniform_distribution" in γ_k_style:
            γ_numerator = client_datasets_size_list[i]
        else:
            γ_numerator = 1
        γ_k = γ_numerator / γ_denominator
        γ_k_list.append(float(γ_k))

    return γ_k_list


def get_j_bar_c_p(y_hat_θ, c, s, p, device):
    y_hat_θ_c = (y_hat_θ == c).to(device)
    s_p = (s == p).to(device)
    joint = (y_hat_θ_c * s_p).to(device)

    P_s_p = (sum(s_p) / len(s)).to(device)  # r_bar(p)
    P_joint = (sum(joint) / len(s)).to(device)
    P_conditional = (P_joint / P_s_p).to(device)  # j_bar(c, p)
    return P_conditional


def get_q_c_p(j_c_p, u_c, r_p, device):
    if j_c_p == 0 or r_p == 0 or u_c == 0:
        q = torch.tensor(0.)
    else:
        q = j_c_p * r_p / torch.sqrt(u_c * r_p)

    return q.to(device)


def get_Q_hat_θ(y_hat_θ, s, device):
    j_bar_0_0 = get_j_bar_c_p(y_hat_θ, 0, s, 0, device)
    j_bar_0_1 = get_j_bar_c_p(y_hat_θ, 0, s, 1, device)
    j_bar_1_0 = get_j_bar_c_p(y_hat_θ, 1, s, 0, device)
    j_bar_1_1 = get_j_bar_c_p(y_hat_θ, 1, s, 1, device)

    u_bar_0 = get_u_bar_c(y_hat_θ, 0, device)
    u_bar_1 = get_u_bar_c(y_hat_θ, 1, device)

    r_bar_0 = get_r_bar_p(s, 0, device)
    r_bar_1 = get_r_bar_p(s, 1, device)

    q_00 = get_q_c_p(j_bar_0_0, u_bar_0, r_bar_0, device)
    q_01 = get_q_c_p(j_bar_0_1, u_bar_0, r_bar_1, device)
    q_10 = get_q_c_p(j_bar_1_0, u_bar_1, r_bar_0, device)
    q_11 = get_q_c_p(j_bar_1_1, u_bar_1, r_bar_1, device)
    Q = torch.tensor([
        [q_00, q_01],
        [q_10, q_11]
    ]).to(device)
    return Q, j_bar_0_0, j_bar_0_1, j_bar_1_0, j_bar_1_1, u_bar_0, u_bar_1


def get_G_hat_θ_hat_v(Q, v, device):
    Q = Q.to(device)
    v = v.reshape(-1, 1).to(device)
    result = v.T.matmul(Q.T).matmul(Q).matmul(v)
    result = result[0][0].to(device)
    result = torch.where(torch.isnan(result), torch.full_like(result, 0), result)
    return result


def get_r_bar_p(s, p, device):
    s_p = (s == p).to(device)
    P_s_p = (sum(s_p) / len(s)).to(device)  # r_bar(p)
    return P_s_p


def get_r_bar_k_p_list(param_dict, num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p):
    r_bar_k_p_list = []
    for k in range(num_clients_K):
        sensitive_attribute = torch.tensor([training_dataset[idx]['protected'] for idx in client_dataset_list[k].indices])

        r_bar_k_p = sum(sensitive_attribute == p) / len(sensitive_attribute)

        r_bar_k_p_list.append(r_bar_k_p)

    return r_bar_k_p_list


def get_statistical_distance(tuple_a, tuple_b):
    # Tuple:  {j_bar_0_0, j_bar_0_1, j_bar_1_0, j_bar_1_1, u_bar_0, u_bar_1, get_parameters(model)}
    j_a = np.array([
        [tuple_a[0].cpu(), tuple_a[1].cpu()],
        [tuple_a[2].cpu(), tuple_a[3].cpu()]
    ])
    j_b = np.array([
        [tuple_b[0].cpu(), tuple_b[1].cpu()],
        [tuple_b[2].cpu(), tuple_b[3].cpu()]
    ])

    u_a = np.array([tuple_a[4].cpu(), tuple_a[5].cpu()])
    u_b = np.array([tuple_b[4].cpu(), tuple_b[5].cpu()])

    θ_a, θ_b = np.array(tuple_a[-1]), np.array(tuple_b[-1])

    j_distance = np.linalg.norm(j_a - j_b)
    u_distance = np.linalg.norm(u_a - u_b)
    try:
        θ_distance = np.linalg.norm(θ_a - θ_b)
    except ValueError:
        differ = (θ_a - θ_b)
        θ_distance = sum([np.linalg.norm(differ[i]) for i in range(len(differ))])
    return j_distance, u_distance, θ_distance


def get_statistical_similarity(tuple_a, tuple_b, λ, ρ):
    j_distance, u_distance, θ_distance = get_statistical_distance(tuple_a, tuple_b)
    # DSim_a_b = θ_distance \
    #            + (-0.5 + (math.sqrt(4 * λ + 1) / 2)) * u_distance \
    #            + (λ + 1 / 2 - (math.sqrt(4 * λ + 1) / 2)) * j_distance
    if θ_distance != 0:
        W_θ_a_b = math.exp(-θ_distance / ρ)
    else:
        W_θ_a_b = 0

    if u_distance != 0:
        W_u_a_b = math.exp(-u_distance / ρ)
    else:
        W_u_a_b = 0

    if j_distance != 0:
        W_j_a_b = math.exp(-j_distance / ρ)
    else:
        W_j_a_b = 0

    return (W_θ_a_b, W_u_a_b, W_j_a_b)

# 用于异步操作的，同步操作可以不管
def get_statistical_tuple(training_dataset, client_dataset_list, client_id, model, mask_s1_flag, hypothesis, device):
    j_bar_0_0_list, j_bar_0_1_list, j_bar_1_0_list, j_bar_1_1_list = [], [], [], []
    u_bar_0_list, u_bar_1_list = [], []

    client_X = torch.stack([training_dataset[idx]['X'] for idx in client_dataset_list[client_id].indices]).to(device)
    local_prediction = model(client_X).to(device)
    if mask_s1_flag:
        s = torch.tensor(
            np.array([training_dataset[idx]['s2'] for idx in client_dataset_list[client_id].indices])).to(device)
    else:
        s = torch.tensor(
            np.array([training_dataset[idx]['s1'] for idx in client_dataset_list[client_id].indices])).to(device)

    if "LR" in hypothesis:
        y_hat_θ = (local_prediction >= 0.5).reshape(-1).to(device)
    else:  # NN
        y_hat_θ = local_prediction.argmax(dim=1).to(device)

    _, temp_j_bar_0_0, temp_j_bar_0_1, temp_j_bar_1_0, temp_j_bar_1_1, \
    temp_u_bar_0, temp_u_bar_1 = get_Q_hat_θ(y_hat_θ, s, device)

    j_bar_0_0_list.append(temp_j_bar_0_0)
    j_bar_0_1_list.append(temp_j_bar_0_1)
    j_bar_1_0_list.append(temp_j_bar_1_0)
    j_bar_1_1_list.append(temp_j_bar_1_1)
    u_bar_0_list.append(temp_u_bar_0)
    u_bar_1_list.append(temp_u_bar_1)

    j_bar_0_0 = sum(j_bar_0_0_list) / len(j_bar_0_0_list)
    j_bar_0_1 = sum(j_bar_0_1_list) / len(j_bar_0_1_list)
    j_bar_1_0 = sum(j_bar_1_0_list) / len(j_bar_1_0_list)
    j_bar_1_1 = sum(j_bar_1_1_list) / len(j_bar_1_1_list)
    u_bar_0 = sum(u_bar_0_list) / len(u_bar_0_list)
    u_bar_1 = sum(u_bar_1_list) / len(u_bar_1_list)

    return (j_bar_0_0, j_bar_0_1, j_bar_1_0, j_bar_1_1, u_bar_0, u_bar_1, np.array(get_parameters(model), dtype=object))


def get_similarity_matrix(tuple_list, λ, ρ):
    K = len(tuple_list)
    similarity_matrix = []
    # 此处可以利用矩阵的对称性进行优化，降低运算次数
    for i in range(K):
        similarity_matrix.append([])
        for j in range(K):
            if i == j:
                similarity_matrix[i].append((1, 1, 1, 1))
            else:
                similarity_matrix[i].append(  # i用户 对 j用户的相似性
                    get_statistical_similarity(tuple_list[i], tuple_list[j], λ, ρ)
                )
    return similarity_matrix


def get_u_bar_c(y_hat_θ, c, device):
    y_hat_θ_c = (y_hat_θ == c).to(device)
    P_y_hat_θ_c = (sum(y_hat_θ_c) / len(y_hat_θ)).to(device)  # u_bar(c)
    return P_y_hat_θ_c


def initialization(param_dict, client_dataset_list, num_clients_K, mask_s1_flag, training_dataset, γ_k_style):
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    global_v = torch.rand(2, 1)

    r_bar_k_p0_list = get_r_bar_k_p_list(param_dict, num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=0)
    r_bar_k_p1_list = get_r_bar_k_p_list(param_dict, num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=1)

    γ_k_list = get_gamma_k_list(γ_k_style, client_datasets_size_list, num_clients_K)

    r_hat_p0 = sum([r_bar_k_p0_list[i] * γ_k_list[i] for i in range(len(γ_k_list))])
    r_hat_p1 = sum([r_bar_k_p1_list[i] * γ_k_list[i] for i in range(len(γ_k_list))])

    v_hat_1 = [math.sqrt(r_hat_p0), math.sqrt(r_hat_p1)]
    return client_datasets_size_list, global_v, r_bar_k_p0_list, r_bar_k_p1_list, γ_k_list, r_hat_p0, r_hat_p1, v_hat_1


def localized_approximation(j_bar_0_0_list, j_bar_0_1_list, j_bar_1_0_list, j_bar_1_1_list,
                            u_bar_0_list, u_bar_1_list,
                            i, local_model_list, similarity_matrix):
    θ_tilde_i = 0
    j_tilde_i_0_0, j_tilde_i_0_1, j_tilde_i_1_0, j_tilde_i_1_1 = 0, 0, 0, 0
    u_tilde_i_0, u_tilde_i_1 = 0, 0
    W_θ_sum, W_u_sum, W_j_sum = 0, 0, 0
    for j in range(len(local_model_list)):
        if j == i:
            continue
        else:
            W_θ, W_u, W_j = similarity_matrix[j][i][0], similarity_matrix[j][i][1], similarity_matrix[j][i][2]
            W_θ_sum += W_θ
            W_u_sum += W_u
            W_j_sum += W_j

            θ_tilde_i += W_θ * np.array(get_parameters(local_model_list[j]))

            j_bar_0_0, j_bar_0_1, j_bar_1_0, j_bar_1_1 = j_bar_0_0_list[j], j_bar_0_1_list[j], j_bar_1_0_list[j], \
                                                         j_bar_1_1_list[j]

            u_bar_0, u_bar_1 = u_bar_0_list[j], u_bar_1_list[j]

            j_tilde_i_0_0 += W_j * j_bar_0_0
            j_tilde_i_0_1 += W_j * j_bar_0_1
            j_tilde_i_1_0 += W_j * j_bar_1_0
            j_tilde_i_1_1 += W_j * j_bar_1_1

            u_tilde_i_0 += W_u * u_bar_0
            u_tilde_i_1 += W_u * u_bar_1

    θ_tilde_i = θ_tilde_i / W_θ_sum

    if W_j_sum == 0:
        j_tilde_i_0_0, j_tilde_i_0_1, j_tilde_i_1_0, j_tilde_i_1_1 = 0,0,0,0
    else:
        j_tilde_i_0_0 = j_tilde_i_0_0 / W_j_sum
        j_tilde_i_0_1 = j_tilde_i_0_1 / W_j_sum
        j_tilde_i_1_0 = j_tilde_i_1_0 / W_j_sum
        j_tilde_i_1_1 = j_tilde_i_1_1 / W_j_sum

    if W_u_sum == 0:
        u_tilde_i_0, u_tilde_i_1 = 0, 0
    else:
        u_tilde_i_0 = u_tilde_i_0 / W_u_sum
        u_tilde_i_1 = u_tilde_i_1 / W_u_sum

    return (j_tilde_i_0_0, j_tilde_i_0_1, j_tilde_i_1_0, j_tilde_i_1_1, u_tilde_i_0, u_tilde_i_1, θ_tilde_i)


def staleness_function_beta_β(zeta_ζ, phi_φ=0.1, function_style="exponential"):
    if "exponential" in function_style:
        compensation = math.exp(-zeta_ζ * phi_φ)
    elif "polynomial" in function_style:
        compensation = math.pow(zeta_ζ, -phi_φ)
    elif "linear" in function_style:
        compensation = phi_φ / zeta_ζ
    else:
        compensation = math.exp(-zeta_ζ * phi_φ)
    return compensation


def Fed_Renyi(device,
            global_model,
            algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
            ):
    # Initialization
    accumulation_steps = int(256 / param_dict['batch_size'])


    mask_s1_flag = False  # 表格型的数据集里面会有2个敏感属性s1和s2，默认是用s1，掩码s2。所以这里默认填False
    logger.info(f'mask_s1_flag: {mask_s1_flag}')

    logger.info("FedRenyi's Initialization")

    try:
        γ_k_style = param_dict['γ_k_style']
        tolerance_τ = param_dict['tolerance_τ']
        lamda = param_dict['FedRenyi_λ']
    except Exception:
        γ_k_style = "uniform_distribution"
        tolerance_τ = communication_round_I * algorithm_epoch_T
        lamda = 1

    client_datasets_size_list, global_v, r_bar_k_p0_list, r_bar_k_p1_list, \
    γ_k_list, r_hat_p0, r_hat_p1, v_hat_1 = initialization(param_dict, client_dataset_list,  num_clients_K,
                                                           mask_s1_flag, training_dataset, γ_k_style)

    training_dataset_size = len(training_dataset.labels)
    client_datasets_size_list = [len(_) for _ in client_dataset_list]

    # basic_path = os.path.join("./save_path", param_dict['dataset_name'],
    #                           param_dict['split_strategy'],
    #                           param_dict['algorithm'],
    #                           param_dict['hypothesis'],
    #                           str(num_clients_K) + "Clients")
    basic_path = param_dict['model_path']

    # Parameter Initialization
    for k in range(param_dict["num_clients_K"]):  # 持久化
        full_path = os.path.join(basic_path, "client_" + str(k + 1), 'model.pt')
        torch.save(global_model, full_path)
    # local_model_list = [copy.deepcopy(global_model) for _ in range(num_clients_K)] # 内存化

    # Training process
    logger.info("Training process begin!")
    logger.info(f'Training Dataset Size: {training_dataset_size}; Client Datasets Size:{client_datasets_size_list}')
    if "SENT_CLF" in param_dict["task"]:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    elif "IMG_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)
    elif "Tabular_CLF" in param_dict["task"]:
        criterion = torch.nn.BCELoss(reduction='none').to(device)

    total_gpu_seconds = 0
    users_gpu_seconds_list = [0] * num_clients_K

    # model_MB_size = sys.getsizeof(global_model.state_dict()) / (1024 ** 2)
    model_MB_size = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)
    # logger.info(f"Model's Communication Cost: {model_MB_size} MB")

    # Simulate Client Parallel
    # TODO:改了迭代的架构，现在有三个for 最外层的for通信轮次 第二层是for每个通信轮次中的客户端训练epoch 第三层是for batch
    for iter_t in range(communication_round_I):
        # Client Selection
        # 先选客户端，只对选中的客戶下发模型
        # FedRenyi需要与所有客户端通信
        idxs_users = [i for i in range(num_clients_K)]

        logger.info(f"Communication Round: {iter_t + 1}; Select clients: {idxs_users}; Start Local Training!")

        # Simulate Client Parallel
        for id in idxs_users:
            # Local Initialization
            # 下发模型
            logger.info("Copy From Global Model")
            model = copy.deepcopy(global_model)
            model.train()
            model.to(device)
            optimizer = BERTCLF_Optimizer(
                method=param_dict['optimize_method'], learning_rate=param_dict['learning_rate'], max_grad_norm=0)
            optimizer.set_parameters(list(model.named_parameters()))
            client_i_dataloader = training_dataloaders[id]

            # Local Training
            for epoch in range(algorithm_epoch_T):
                # 设置状态变量
                epoch_total_loss = 0
                epoch_total_size = 0

                # 注意：mini-batch gradient descent一般是把整个batch的损失累加起来，然后除以batch内的样本数目
                # FedAvg算法中，一个batch就更新一次参数
                for batch_id, batch in enumerate(client_i_dataloader):
                # for batch in client_i_dataloader:
                    if "SENT_CLF" in param_dict["task"]:
                        # input_ids尺寸 [batch_size, max_len]
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        protected = batch["protected"].to(device)
                    elif "IMG_CLF" in param_dict["task"]:
                        imgs = batch["img"].to(device)
                        protected = batch["protected"].to(device)
                    elif "Tabular_CLF" in param_dict["task"]:
                        X = batch["X"].to(device)
                        # s = batch["s2"] if mask_s1_flag else batch["s1"]
                        # protected = s
                        protected = batch['protected'].to(device)
                    # labels尺寸 [batch_size]
                    labels = batch["labels"].to(device)

                    # 考虑到有可能没取满一整个batch，所以动态获取一下实际batch_size
                    true_batch_size = labels.size()[0]
                    epoch_total_size += true_batch_size

                    # 记录GPU计算开始时间
                    gpu_start_time = time.time()

                    if "SENT_CLF" in param_dict["task"]:
                        # features尺寸 [batch_size, emb_dim]
                        # logits尺寸 [batch_size, category]
                        features, logits = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        # activated_preds = logits.softmax(dim=1)
                        activated_preds = logits  # 由于我们采用了torch.nn.CrossEntropyLoss，在Pytorch里面这个函数是已经加了softmax的，所以我们不需要再手动加softmax
                        _, preds = torch.max(activated_preds, dim=1)
                        # batch_loss尺寸 [batch_size]
                        batch_loss = criterion(activated_preds, labels)

                    elif "IMG_CLF" in param_dict["task"]:
                        # preds尺寸 [batch_size, 1]
                        # features尺寸 [batch_size, emb_dim]
                        preds, features = model(imgs)
                        batch_loss = criterion(preds[:, 0], labels.float())

                    elif "Tabular_CLF" in param_dict["task"]:
                        # local_prediction尺寸 [batch_size, 1]
                        if "ANN" in str(type(model)):
                            preds, features = model(X)
                        elif "LogisticRegression" in str(type(model)):
                            preds = model(X)
                        else:
                            preds = model(X)
                        batch_loss = criterion(preds[:, 0], labels.float())

                    y_hat_θ = (preds >= 0.5).reshape(-1).to(device)

                    Q, _, _, _, _, _, _ = get_Q_hat_θ(y_hat_θ, protected, device)
                    G = get_G_hat_θ_hat_v(Q, global_v, device).to(device)
                    regularization_term = lamda * G

                    loss = torch.sum(batch_loss) / true_batch_size
                    loss += regularization_term


                    loss.backward()


                    if (batch_id + 1) % accumulation_steps == 0:
                        # FedAvg算法一个batch就做一次更新
                        optimizer.step()

                    # 记录GPU计算结束时间
                    gpu_end_time = time.time()

                    users_gpu_seconds_list[id] += (gpu_end_time - gpu_start_time)

                    # 清空梯度
                    model.zero_grad()
                    # 记录状态信息
                    epoch_total_loss += loss
                    # average_one_sample_loss_in_epoch += average_one_sample_loss_in_batch / math.ceil(
                    #     client_datasets_size_list[id] / param_dict['batch_size'])

                    if "SENT_CLF" in param_dict["task"]:
                        del input_ids, attention_mask, labels
                    elif "IMG_CLF" in param_dict["task"]:
                        del imgs, labels

                    gc.collect()

                    # if param_dict["one_batch_per_Epoch"]:
                    #     break

                average_one_sample_loss_in_epoch = epoch_total_loss / epoch_total_size
                logger.info(f"Communication Round: {iter_t + 1} / {communication_round_I}; "
                            f"Client: {id} / {num_clients_K}; "
                            f"Epoch: {epoch + 1}; Avg One Sample's Loss Over Epoch: {average_one_sample_loss_in_epoch}")

            # Upgrade the local model list
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            # local_model_list[id] = model.cpu()  # 内存化
            torch.save(model.cpu(), client_model_path)  # 持久化

            del model
            gc.collect()
            # torch.cuda.empty_cache()

        # Communicate
        total_gpu_seconds += sum(users_gpu_seconds_list)
        logger.info(
            f"Communication Round {(iter_t + 1)} 's Communication Cost: {(iter_t + 1) * len(idxs_users) * 2 * model_MB_size} MB")

        # Global operation
        logger.info("Parameter aggregation")
        theta_list = []
        client_model_path_list = []
        for id in idxs_users:
            client_model_path = os.path.join(basic_path, "client_" + str(id + 1), 'model.pt')
            selected_model = torch.load(client_model_path, weights_only=False)  # 持久化
            client_model_path_list.append(client_model_path)
            theta_list.append(get_parameters(selected_model))
            del selected_model
            gc.collect()

        theta_list = np.array(theta_list, dtype=object)
        # FedAvg新版论文的聚合权重是数据占比
        # 这个地方要自己去验证一下np.average的加权平均的用法，有点反直觉的，weights参数只需要传权重的“分子”，不用传整个分数，“分母”会自动除
        # 如一个weights = [w1, w2, w3, w4]
        # 那么结果就是(theta1 * w1 + theta2 * w2 + theta3 * w3 + theta4 * w4)/ sum(w1+w2+w3+w4)
        theta_avg = np.average(theta_list, axis=0, weights=[client_datasets_size_list[j] for j in idxs_users]).tolist()
        # FedAvg旧版论文的聚合权重是平均
        # theta_avg = np.mean(theta_list, 0).tolist()

        # 检查theta_avg的结果是否存在非np.ndarray的项，如有则转换
        # checked_theta_avg = []
        # for item in theta_avg:
        #     if isinstance(item, np.ndarray):
        #         checked_theta_avg.append(item)
        #     elif isinstance(item, np.float64):
        #         checked_theta_avg.append(np.array([item]))
        # theta_avg = checked_theta_avg

        logger.info("Update Global Model")
        set_parameters(global_model, theta_avg)

        logger.info("********** Global v update **********")
        backup_v = global_v
        try:
            global_v = get_argmax_v(param_dict, [i for i in range(num_clients_K)], client_model_path_list, mask_s1_flag,
                                    training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, device, γ_k_style)
        except Exception:
            global_v = backup_v



        # 当前消耗的总GPU秒，平均GPU秒
        avg_gpu_seconds = (total_gpu_seconds / num_clients_K)
        logger.info(
            f"Global Model testing at Communication {(iter_t + 1)}/ {communication_round_I}")
        logger.info(
            f"Total GPU seconds: {total_gpu_seconds}, Avg GPU seconds over client: {avg_gpu_seconds}")

        del theta_list
        gc.collect()

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


    logger.info("Training finish, save and return the global model.")
    # Save global model
    save_dir = f'./save_path/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"global_FedRenyi_"+γ_k_style+".pt")
    torch.save(global_model, save_path)
    total_communication_cost = communication_round_I * num_clients_K * FL_fraction * 2 * model_MB_size
    return global_model, total_gpu_seconds, total_communication_cost

