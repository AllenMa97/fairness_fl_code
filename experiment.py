import gc
import os
import sys
import torch
import copy
import numpy as np
import time
import multiprocessing as mp
from functools import partial

from tool.logger import *
from tool.utils import check_and_make_the_path, FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import check_resume_status, save_checkpoint, load_checkpoint
from moudle.experiment_setup import Experiment_Create_dataset, Experiment_Create_dataloader, Experiment_Create_model
from algorithm.SeparateTraining import ST_BertClassifier
from algorithm.FederatedAverage import Fed_AVG
from algorithm.FederatedProximal import Fed_Prox
from algorithm.Scaffold import Scaffold
from algorithm.FederatedNova import Fed_Nova
from algorithm.FederatedRep import Fed_Rep
from algorithm.FederatedProto import Fed_PROTO
from algorithm.OSFL import OneShotFed
from algorithm.CoBoosting import Co_Boosting
from algorithm.FairFed import FairFed
from algorithm.FedFair import FedFair
from algorithm.FL_FairBatch import FL_FairBatch
from algorithm.FedFB import FedFB
from algorithm.FederatedRenyi import Fed_Renyi
from algorithm.DOSFL import DistilledOneShotFed
# from algorithm.abandon.PoTrain import PoTrain
from algorithm.NaiveMix import NaiveMix
from algorithm.FedMix import FedMix
from algorithm.mFairFL import mFairFL
from algorithm.PDFFed import PDF_Fed
from algorithm.PraFFL import PraFFL
from algorithm.FedFACT import FedFACT
from algorithm.LoGoFair import LoGoFair
from algorithm.backup.DENSE import DENSE
from algorithm.backup.FENS import FENS
from algorithm.backup.FedCAV import FedCAV
from algorithm.backup.FedDEO import FedDEO
from algorithm.backup.FedELMY import FedELMY
from algorithm.backup.FedFisher import FedFisher
from algorithm.backup.FedKD import FedKD
from ablation.PDFFed_Abl import *
from ablation.PDFFed_V2_Abl import *


def calculate_communication_cost(algorithm_name, param_dict, global_model):
    I = param_dict['communication_round_I']
    K = param_dict['num_clients_K']
    fraction = param_dict['FL_fraction']
    task = param_dict.get('task', '')

    model_MB = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)

    if "SENT_CLF" in task:
        emb_dim = param_dict.get('emb_dim', 768)
        rep_MB = sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024 * 1024)
        clf_params_count = sum(p.numel() for p in global_model.out.parameters())
    elif "IMG_CLF" in task:
        emb_dim = param_dict.get('emb_dim', 512)
        rep_MB = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_params_count = sum(p.numel() for p in global_model.out_layer.parameters())
    elif "Tabular_CLF" in task:
        emb_dim = param_dict.get('emb_dim', param_dict.get('nn_input_size', 128))
        rep_MB = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_params_count = sum(p.numel() for p in global_model.out_layer.parameters())
    else:
        emb_dim = param_dict.get('emb_dim', 768)
        rep_MB = model_MB
        clf_params_count = 0

    num_of_class = 2
    prototype_MB = num_of_class * emb_dim * 4 / (1024 * 1024)
    group_prototype_MB = 4 * emb_dim * 4 / (1024 * 1024)

    selected_per_round = K * fraction

    cost = 0.0

    # ---- 标准 FL: 上传模型 + 下载模型 ----
    if algorithm_name in ["Fed_AVG", "Fed_Prox", "Fed_Nova", "FedFB",
                           "FL_FairBatch", "LoGoFair", "mFairFL", "ProxProbability"]:
        cost = I * selected_per_round * 2 * model_MB

    # ---- FedRenyi: 全部客户端参与(不采样) ----
    elif algorithm_name == "Fed_Renyi":
        cost = I * K * 2 * model_MB

    # ---- Scaffold: 上传delta_y+delta_c(各=model大小), 下载model+c(各=model大小) = 4x ----
    elif algorithm_name == "Scaffold":
        cost = I * selected_per_round * 4 * model_MB

    # ---- FedProto: 下载model, 上传prototype(不上传模型参数) ----
    elif algorithm_name == "Fed_PROTO":
        cost = I * selected_per_round * (model_MB + prototype_MB)

    # ---- FedRep(论文): 仅通信representation层, classifier head保留本地 ----
    elif algorithm_name == "Fed_Rep":
        cost = I * selected_per_round * 2 * rep_MB

    # ---- PDFFed: 上传model+4组群组原型, 下载model ----
    elif algorithm_name == "PDF_Fed":
        cost = I * selected_per_round * (2 * model_MB + group_prototype_MB)

    # ---- FairFed: 下载model, 上传标量损失(可忽略) ----
    elif algorithm_name == "FairFed":
        cost = I * selected_per_round * model_MB

    # ---- NaiveMix/FedMix: 标准模型通信 + 全部客户端上传Mash数据 ----
    elif algorithm_name in ["NaiveMix", "FedMix"]:
        cost = I * selected_per_round * 2 * model_MB
        mash_MB_per_client = (emb_dim + 1) * 4 / (1024 * 1024)
        cost += I * K * mash_MB_per_client

    # ---- OSFL/CoBoosting: 单轮, 全部客户端 ----
    elif algorithm_name in ["OneShotFed", "Co_Boosting"]:
        cost = K * 2 * model_MB

    # ---- DOSFL: 单轮, 全部客户端, 下载model, 上传蒸馏数据(不上传模型) ----
    elif algorithm_name == "DistilledOneShotFed":
        Sd = 5
        max_len = param_dict.get('max_len', 128)
        cost = K * model_MB
        cost += K * Sd * max_len * emb_dim * 4 / (1024 * 1024)
        cost += K * Sd * max_len * 2 * 4 / (1024 * 1024)
        cost += K * Sd * 4 / (1024 * 1024)

    # ---- PraFFL: 上传/下载 model + HyperNetwork(远小于model) ----
    elif algorithm_name == "PraFFL":
        hidden_dim = param_dict.get('hypernet_hidden', 256)
        hypernet_params = (1 * hidden_dim + hidden_dim) + (hidden_dim * hidden_dim + hidden_dim) + (hidden_dim * clf_params_count + clf_params_count)
        hypernet_MB = hypernet_params * 4 / (1024 * 1024)
        cost = I * selected_per_round * 2 * (model_MB + hypernet_MB)

    # ---- FedFACT: 下载2份model(ensemble), 上传1份model ----
    elif algorithm_name == "FedFACT":
        cost = I * selected_per_round * 3 * model_MB

    else:
        cost = I * selected_per_round * 2 * model_MB

    return round(cost, 3)

def _cleanup_intermediate_models(model_path, logger):
    import shutil
    if not os.path.exists(model_path):
        return
    client_dirs = [d for d in os.listdir(model_path) if d.startswith("client_")]
    for cd in client_dirs:
        cd_path = os.path.join(model_path, cd)
        if os.path.isdir(cd_path):
            shutil.rmtree(cd_path)
            logger.info(f"[Cleanup] Removed client model dir: {cd}")
    step_global_files = [f for f in os.listdir(model_path) if f.startswith("step_") and f.endswith(".pt")]
    final_global_files = [f for f in os.listdir(model_path) if f.startswith("final_") and f.endswith(".pt")]
    if len(final_global_files) > 1:
        final_global_files.sort(key=lambda x: int(x.split("_")[1]))
        for f in final_global_files[:-1]:
            os.remove(os.path.join(model_path, f))
            logger.info(f"[Cleanup] Removed intermediate global model: {f}")


def Experiment_SeparateTraining(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                testing_dataloader, testing_dataset):
    device = param_dict['device']
    acc_list, DEO_list, SPD_list = [], [], []
    testing_dataset_len = len(testing_dataset)

    for time in range(3):
        # 训练并持久化
        ST_BertClassifier(
            device,
            global_model,
            param_dict['algorithm_epoch_T'],
            param_dict['num_clients_K'],
            param_dict['communication_round_I'],
            param_dict['FL_fraction'],
            param_dict['FL_drop_rate'],
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            param_dict,
            testing_dataloader,
            testing_dataset_len
        )


        # 测试
        logger.info("Client models testing")
        accuracy, DEO, SPD = FL_fairness_and_accuracy_test(param_dict, testing_dataloader, testing_dataset_len)
        acc_list.append(accuracy)
        DEO_list.append(DEO)
        SPD_list.append(SPD)

    acc_list_mean, acc_list_std = round(float(np.mean(np.array(acc_list))), 3), round(float(np.std(np.array(acc_list))), 3)
    DEO_list_mean, DEO_list_std = round(float(np.mean(np.array(DEO_list))), 3), round(float(np.std(np.array(DEO_list))), 3)
    SPD_list_mean, SPD_list_std = round(float(np.mean(np.array(SPD_list))), 3), round(float(np.std(np.array(SPD_list))), 3)
    logger.info(f"****** ACC Mean±STD: {acc_list_mean}+'±'+{acc_list_std} ******")
    logger.info(f"****** DEO Mean±STD: {DEO_list_mean}+'±'+{DEO_list_std} ******")
    logger.info(f"****** SPD Mean±STD: {SPD_list_mean}+'±'+{SPD_list_std} ******")

    with open(param_dict['result_path'], 'a+', encoding='utf-8') as f:
        f.write("ACC Mean±STD: " + str(acc_list_mean) + "±" + str(acc_list_std) + '\n')
        f.write("DEO Mean±STD: " + str(DEO_list_mean) + "±" + str(DEO_list_std) + '\n')
        f.write("SPD Mean±STD: " + str(SPD_list_mean) + "±" + str(SPD_list_std) + '\n')
        f.write("----------------------------------------------------------------------------\n")

    _cleanup_intermediate_models(param_dict['model_path'], logger)

    try:
        from tool.notification import notify_experiment_done
        notify_experiment_done(
            algorithm=param_dict['algorithm'],
            dataset=param_dict.get('dataset_name', param_dict.get('dataset', '')),
            result_path=param_dict['result_path'],
            extra_info=f"Split: {param_dict.get('split_strategy', '')}, Clients: {param_dict.get('num_clients_K', '')}, "
                       f"ACC: {acc_list_mean}±{acc_list_std}, DEO: {DEO_list_mean}±{DEO_list_std}"
        )
    except Exception:
        pass


def _run_single_repeat(repeat_idx, algorithm_function, param_dict, global_model_state,
                        training_dataloaders, training_dataset, client_dataset_list,
                        testing_dataloader, testing_dataset, formula_comm_cost):
    """在独立进程中运行单次 repeat 实验，返回测试结果字典"""
    import torch
    import copy
    import gc
    import os
    import time
    import numpy as np
    from tool.logger import setup_logger
    from tool.utils import (FL_fairness_and_accuracy_test,
                            FL_fairness_and_accuracy_test_4_IMG_CLF,
                            FL_fairness_and_accuracy_test_4_Tabular_CLF,
                            get_HM_by_two_value)
    from tool.checkpoint import check_resume_status, save_checkpoint, load_checkpoint

    # 为每个 repeat 设置独立的日志文件
    repeat_param_dict = dict(param_dict)
    repeat_param_dict['Experiment_NO'] = repeat_idx + 1
    log_path = repeat_param_dict['log_path']
    result_path = repeat_param_dict['result_path']

    logger = setup_logger(log_path)

    device = repeat_param_dict['device']
    testing_dataset_len = len(testing_dataset)

    logger.info(f"****** Now Playing the {(repeat_idx+1)}-th / 3 experiment for more persuasive Result! ******")

    # 构建模型
    from moudle.experiment_setup import Experiment_Create_model
    global_model = Experiment_Create_model(repeat_param_dict)
    global_model.load_state_dict(global_model_state)

    # 检查断点恢复
    checkpoint = check_resume_status(repeat_param_dict)
    resume_round = checkpoint['communication_round'] + 1 if checkpoint else 0

    if checkpoint:
        global_model.load_state_dict(checkpoint['global_model_state'])
        logger.info(f"****** Resuming from round {resume_round} ******")

    # 训练
    trained_global_model, trained_gpu_seconds, trained_communication_cost = algorithm_function(
        device,
        global_model,
        repeat_param_dict['algorithm_epoch_T'],
        repeat_param_dict['num_clients_K'],
        repeat_param_dict['communication_round_I'],
        repeat_param_dict['FL_fraction'],
        repeat_param_dict['FL_drop_rate'],
        training_dataloaders,
        training_dataset,
        client_dataset_list,
        repeat_param_dict,
        testing_dataloader,
        testing_dataset_len,
        start_round=resume_round
    )

    # 测试
    logger.info(f"****** Trained Global Model Testing ******")
    result = {}
    if "SENT_CLF" in repeat_param_dict["task"]:
        accuracy, DEO, SPD = FL_fairness_and_accuracy_test(trained_global_model, repeat_param_dict, testing_dataloader, testing_dataset_len)
        logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)}")
    elif "IMG_CLF" in repeat_param_dict["task"]:
        accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(trained_global_model, repeat_param_dict, testing_dataloader, testing_dataset_len)
        FR = 1 - DEO
        HM = get_HM_by_two_value(accuracy, FR)
        logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        result['FR'] = float(FR)
        result['HM'] = float(HM)
    elif "Tabular_CLF" in repeat_param_dict["task"]:
        accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(trained_global_model, repeat_param_dict,
                                                                     testing_dataloader, testing_dataset_len)
        FR = 1 - DEO
        HM = get_HM_by_two_value(accuracy, FR)
        logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                    f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
        result['FR'] = float(FR)
        result['HM'] = float(HM)

    result['ACC'] = float(accuracy)
    result['DEO'] = float(DEO)
    result['SPD'] = float(SPD)
    result['gpu_seconds'] = float(trained_gpu_seconds)
    result['communication_cost'] = float(formula_comm_cost)

    del trained_global_model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def Experiment_FL(algorithm_function, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader, testing_dataset):
    device = param_dict['device']
    testing_dataset_len = len(testing_dataset)
    exp_repeat_times = param_dict.get('exp_repeat_times', 3)
    parallel_repeats = param_dict.get('parallel_repeats', 1)
    parallel_repeats = max(1, min(parallel_repeats, exp_repeat_times))

    formula_comm_cost = calculate_communication_cost(algorithm_function.__name__, param_dict, global_model)
    logger.info(f"****** Communication Cost (Formula): {formula_comm_cost} MB ******")

    # 收集需要跑的 repeat 索引
    # repeat_checkpoints: 存储不完整但可从断点恢复的 repeat
    repeat_indices = []
    repeat_checkpoints = {}
    total_rounds = param_dict.get('communication_round_I', 0)
    for t in range(exp_repeat_times):
        repeat_param = dict(param_dict)
        repeat_param['Experiment_NO'] = t + 1
        checkpoint = load_checkpoint(repeat_param)
        if checkpoint is not None and 'global_model_state' in checkpoint:
            current_round = checkpoint['communication_round']
            if current_round >= total_rounds - 1:
                # 已完成的 repeat → 跳过
                logger.info(f"****** Repeat {t+1}/{exp_repeat_times} already completed (round {current_round+1}/{total_rounds}), skipping ******")
                continue
            else:
                # 不完整的 repeat → 标记为可从断点恢复
                logger.info(f"****** Repeat {t+1}/{exp_repeat_times} incomplete (round {current_round+1}/{total_rounds}), will resume ******")
                repeat_checkpoints[t] = checkpoint
        repeat_indices.append(t)

    if not repeat_indices:
        logger.info("****** All repeats already completed, skipping ******")
        return

    global_model_state = copy.deepcopy(global_model).state_dict()

    if parallel_repeats > 1 and len(repeat_indices) > 1:
        # ===== 并行模式 =====
        actual_workers = min(parallel_repeats, len(repeat_indices))
        logger.info(f"****** Running {len(repeat_indices)} repeats with {actual_workers} parallel workers ******")

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=actual_workers) as pool:
            args_list = [
                (idx, algorithm_function, param_dict, global_model_state,
                 training_dataloaders, training_dataset, client_dataset_list,
                 testing_dataloader, testing_dataset, formula_comm_cost)
                for idx in repeat_indices
            ]
            results = pool.starmap(_run_single_repeat, args_list)

        # 收集结果
        acc_list, DEO_list, SPD_list = [], [], []
        FR_list, HM_list = [], []
        gpu_seconds_list, communication_cost_list = [], []
        for r in results:
            acc_list.append(r['ACC'])
            DEO_list.append(r['DEO'])
            SPD_list.append(r['SPD'])
            gpu_seconds_list.append(r['gpu_seconds'])
            communication_cost_list.append(r['communication_cost'])
            if 'FR' in r:
                FR_list.append(r['FR'])
                HM_list.append(r['HM'])
    else:
        # ===== 串行模式（原始逻辑） =====
        acc_list, DEO_list, SPD_list = [], [], []
        FR_list, HM_list = [], []
        gpu_seconds_list, communication_cost_list = [], []

        for t in repeat_indices:
            torch.cuda.empty_cache()
            global_model_backup = copy.deepcopy(global_model)

            repeat_param = dict(param_dict)
            repeat_param['Experiment_NO'] = t + 1
            checkpoint = repeat_checkpoints.get(t)
            resume_round = checkpoint['communication_round'] + 1 if checkpoint else 0

            logger.info(f"****** Now Playing the {(t+1)}-th / {exp_repeat_times} experiment for more persuasive Result! ******")

            if checkpoint:
                global_model_backup.load_state_dict(checkpoint['global_model_state'])
                logger.info(f"****** Resuming from round {resume_round} ******")

            trained_global_model, trained_gpu_seconds, trained_communication_cost = algorithm_function(
                device,
                global_model_backup,
                param_dict['algorithm_epoch_T'],
                param_dict['num_clients_K'],
                param_dict['communication_round_I'],
                param_dict['FL_fraction'],
                param_dict['FL_drop_rate'],
                training_dataloaders,
                training_dataset,
                client_dataset_list,
                param_dict,
                testing_dataloader,
                testing_dataset_len,
                start_round=resume_round
            )
            # 测试
            logger.info(f"****** Trained Global Model Testing ******")
            if "SENT_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test(trained_global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                FR_list.append(float(FR))
                HM_list.append(float(HM))
            elif "IMG_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(trained_global_model, param_dict, testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                FR_list.append(float(FR))
                HM_list.append(float(HM))
            elif "Tabular_CLF" in param_dict["task"]:
                accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(trained_global_model, param_dict,
                                                                             testing_dataloader, testing_dataset_len)
                FR = 1 - DEO
                HM = get_HM_by_two_value(accuracy, FR)
                logger.info(f"ACC: {round(float(accuracy), 3)}, DEO: {round(float(DEO), 3)}, SPD:{round(float(SPD), 3)},"
                            f" FR: {round(float(FR), 3)}, HM: {round(float(HM), 3)}")
                FR_list.append(float(FR))
                HM_list.append(float(HM))

            acc_list.append(float(accuracy))
            DEO_list.append(float(DEO))
            SPD_list.append(float(SPD))
            gpu_seconds_list.append(float(trained_gpu_seconds))
            communication_cost_list.append(formula_comm_cost)

            del trained_global_model
            gc.collect()
            torch.cuda.empty_cache()

    # 汇总结果
    acc_list_mean, acc_list_std = round(float(np.mean(np.array(acc_list))), 3), round(float(np.std(np.array(acc_list))), 3)
    DEO_list_mean, DEO_list_std = round(float(np.mean(np.array(DEO_list))), 3), round(float(np.std(np.array(DEO_list))), 3)
    SPD_list_mean, SPD_list_std = round(float(np.mean(np.array(SPD_list))), 3), round(float(np.std(np.array(SPD_list))), 3)
    gpu_seconds_list_mean, gpu_seconds_list_std = round(float(np.mean(np.array(gpu_seconds_list))), 3), round(float(np.std(np.array(gpu_seconds_list))), 3)
    communication_cost_list_mean, communication_cost_list_std = round(float(np.mean(np.array(communication_cost_list))), 3), round(float(np.std(np.array(communication_cost_list))), 3)
    logger.info(f"****** {algorithm_function.__name__} ACC Mean±STD: {acc_list_mean}±{acc_list_std} ******")
    logger.info(f"****** {algorithm_function.__name__} DEO Mean±STD: {DEO_list_mean}±{DEO_list_std} ******")
    logger.info(f"****** {algorithm_function.__name__} SPD Mean±STD: {SPD_list_mean}±{SPD_list_std} ******")
    logger.info(f"****** {algorithm_function.__name__} Gpu Seconds Mean±STD: {gpu_seconds_list_mean}±{gpu_seconds_list_std} ******")
    logger.info(f"****** {algorithm_function.__name__} Communication Cost Mean±STD: {communication_cost_list_mean}±{communication_cost_list_std} ******")
    if "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"] or "SENT_CLF" in param_dict["task"]:
        if FR_list:
            FR_list_mean, FR_list_std = round(float(np.mean(np.array(FR_list))), 3), round(float(np.std(np.array(FR_list))), 3)
            HM_list_mean, HM_list_std = round(float(np.mean(np.array(HM_list))), 3), round(float(np.std(np.array(HM_list))), 3)
            logger.info(f"****** {algorithm_function.__name__} FR Mean±STD: {FR_list_mean}±{FR_list_std} ******")
            logger.info(f"****** {algorithm_function.__name__} HM Mean±STD: {HM_list_mean}±{HM_list_std} ******")

    with open(param_dict['result_path'], 'a+', encoding='utf-8') as f:
        f.write(algorithm_function.__name__ +" ACC Mean±STD: " + str(acc_list_mean) + "±" + str(acc_list_std) + '\n')
        f.write(algorithm_function.__name__ +" DEO Mean±STD: " + str(DEO_list_mean) + "±" + str(DEO_list_std) + '\n')
        f.write(algorithm_function.__name__ +" SPD Mean±STD: " + str(SPD_list_mean) + "±" + str(SPD_list_std) + '\n')
        f.write(algorithm_function.__name__ +" Gpu Seconds Mean±STD: " + str(gpu_seconds_list_mean) + "±" + str(gpu_seconds_list_std) + '\n')
        f.write(algorithm_function.__name__ +" Communication Cost Mean±STD: " + str(communication_cost_list_mean) + "±" + str(communication_cost_list_std) + '\n')
        if "IMG_CLF" in param_dict["task"] or "Tabular_CLF" in param_dict["task"] or "SENT_CLF" in param_dict["task"]:
            if FR_list:
                f.write(algorithm_function.__name__ +" FR Mean±STD: " + str(FR_list_mean) + "±" + str(FR_list_std) + '\n')
                f.write(algorithm_function.__name__ +" HM Mean±STD: " + str(HM_list_mean) + "±" + str(HM_list_std) + '\n')

        f.write("----------------------------------------------------------------------------\n")

    _cleanup_intermediate_models(param_dict['model_path'], logger)



def Experiment_pFL(algorithm_function, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader, testing_dataset):
    pass


def Experiment(param_dict):
    # 统一 AMP 控制：根据 GPU 能力自动决定是否启用混合精度
    from tool.amp_utils import resolve_amp_config
    param_dict['use_amp'] = resolve_amp_config(param_dict)

    # # Create dataset
    logger.info("Creating dataset")
    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(param_dict)

    # Create dataloader
    logger.info("Creating dataloader")
    training_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(
        param_dict, training_dataset, validation_dataset, testing_dataset, param_dict['split_strategy'])

    # Model Construction
    # 为了避免过多的随机性影响，尽量保证在同一个初始的模型开始训练
    global_init_model_dir = r"./save_path/" + param_dict['dataset']
    check_and_make_the_path(global_init_model_dir)
    global_model = Experiment_Create_model(param_dict)

    if "SENT_CLF" in param_dict["task"]:
        global_init_model_path = global_init_model_dir+"/global_model_init.pt"
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            try:
                global_model.load_state_dict(torch.load(global_init_model_path, weights_only=False).state_dict())
            except Exception as e:
                logger.error(e)
        global_model.bert.finetune = True
        global_model.out.finetune = True
    elif "IMG_CLF" in param_dict["task"]:
        global_init_model_path = global_init_model_dir + "/global_model_4_IMG_CLF_init.pt"
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            try:
                global_model.load_state_dict(torch.load(global_init_model_path, weights_only=False).state_dict())
            except Exception as e:
                logger.error(e)
    elif "Tabular_CLF" in param_dict["task"]:
        global_init_model_path = global_init_model_dir + "/global_model_4_Tabular_CLF_init.pt"
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            try:
                global_model.load_state_dict(torch.load(global_init_model_path, weights_only=False).state_dict())
            except Exception as e:
                logger.error(e)

    # SeparateTraining
    if ("Separate" in param_dict["algorithm"]) or ("separate" in param_dict["algorithm"]) or (
            "sepa" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: SeparateTraining ~~~~~~")
        Experiment_SeparateTraining(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader,testing_dataset
        )
    # CentralizedTraining
    elif ("Centralized" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: CentralizedTraining ~~~~~~")
        Experiment_SeparateTraining(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader,testing_dataset

    )
    # Federated Average
    elif ("FederatedAverage" in param_dict["algorithm"]) or ("FedAvg" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Average ~~~~~~")
        Experiment_FL(
            Fed_AVG, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader,testing_dataset
        )
    # Federated Prox
    elif ("FederatedProximal" in param_dict["algorithm"]) or ("FedProx" in param_dict["algorithm"]) or (
            "fedprox" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Proximal ~~~~~~")
        Experiment_FL(Fed_Prox, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                  testing_dataloader,testing_dataset)


    # SCAFFOLD
    elif ("Scaffold" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Scaffold ~~~~~~")
        Experiment_FL(Scaffold, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                            testing_dataloader,testing_dataset)


    # Federated Nova
    elif ("FederatedNova" in param_dict["algorithm"]) or ("FedNova" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Nova ~~~~~~")
        Experiment_FL(
            Fed_Nova, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
            testing_dataloader,testing_dataset
        )

    # FedRep
    elif ("FedRep" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Rep ~~~~~~")
        Experiment_FL(Fed_Rep, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader,testing_dataset)

    # FedProto
    elif ("FedProto" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Proto ~~~~~~")
        Experiment_FL(Fed_PROTO, param_dict, global_model, training_dataloaders, training_dataset,
                                 client_dataset_list, testing_dataloader,testing_dataset)

    # One-Shot Federated Learning
    elif ("OSFL" in param_dict["algorithm"]) and (param_dict["algorithm"] != "DOSFL"):
        logger.info("~~~~~~ Algorithm: One-Shot Federated Learning ~~~~~~")
        Experiment_FL(OneShotFed, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # CO_BOOSTING
    elif ("CO_BOOSTING" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: CO-BOOSTING ~~~~~~")
        Experiment_FL(Co_Boosting, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # DOSFL
    elif ("DistilledOneShotFed" in param_dict["algorithm"]) or ("DOSFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: DOSFL ~~~~~~")
        Experiment_FL(DistilledOneShotFed, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FedFair
    elif ("FedFair" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFair ~~~~~~")
        Experiment_FL(FedFair, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FL_FairBatch
    elif ("FL_FairBatch" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FL_FairBatch ~~~~~~")
        Experiment_FL(FL_FairBatch, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FedFB
    elif ("FedFB" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFB ~~~~~~")
        Experiment_FL(FedFB, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FairFed
    elif ("FairFed" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FairFed ~~~~~~")
        Experiment_FL(FairFed, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FedRenyi
    elif ("FedRenyi" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedRenyi ~~~~~~")
        Experiment_FL(Fed_Renyi, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FedMix
    elif ("FedMix" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedMix ~~~~~~")
        Experiment_FL(FedMix, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # NaiveMix
    elif ("NaiveMix" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: NaiveMix ~~~~~~")
        Experiment_FL(NaiveMix, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # mFairFL
    elif ("mFairFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: mFairFL ~~~~~~")
        Experiment_FL(mFairFL, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # PDFFed
    elif ("PDFFed" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: PDFFed ~~~~~~")
        Experiment_FL(PDF_Fed, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # PraFFL (KDD 2025)
    elif ("PraFFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: PraFFL ~~~~~~")
        Experiment_FL(PraFFL, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FedFACT (NeurIPS 2025)
    elif ("FedFACT" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFACT ~~~~~~")
        Experiment_FL(FedFACT, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # LoGoFair
    elif ("LoGoFair" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: LoGoFair ~~~~~~")
        Experiment_FL(LoGoFair, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # DENSE (NeurIPS 2022)
    elif ("DENSE" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: DENSE ~~~~~~")
        Experiment_FL(DENSE, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset_len)

    # FENS (NeurIPS 2024)
    elif ("FENS" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FENS ~~~~~~")
        Experiment_FL(FENS, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset_len)

    # FedCAV (ICLR 2023)
    elif ("FedCAV" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedCAV ~~~~~~")
        Experiment_FL(FedCAV, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset_len)

    # FedDEO (ACM MM 2024)
    elif ("FedDEO" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedDEO ~~~~~~")
        Experiment_FL(FedDEO, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset_len)

    # FedELMY (ACM MM 2024)
    elif ("FedELMY" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedELMY ~~~~~~")
        Experiment_FL(FedELMY, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset_len)

    # FedFisher (AISTATS 2024)
    elif ("FedFisher" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFisher ~~~~~~")
        Experiment_FL(FedFisher, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset_len)

    # FedKD (AAAI 2022)
    elif ("FedKD" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedKD ~~~~~~")
        Experiment_FL(FedKD, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset_len)

    else:
        raise ValueError(f'''Wrong algorithm name:{param_dict['algorithm']} It should be in the following type:
            [Separate | FedAvg | FedProx | Scaffold | FederatedNova | FedRep | FedProto| OSFL | CO_BOOSTING | DOSFL |
             FedFair | FL_FairBatch | FedFB | FairFed | mFairFL | PDFFed | PraFFL | FedFACT |
             DENSE | FENS | FedCAV | FedDEO | FedELMY | FedFisher | FedKD] ''')


def PDFFed_Ablation_Experiment(param_dict):
    # # Create dataset
    logger.info("Creating dataset")
    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(param_dict)

    # Create dataloader
    logger.info("Creating dataloader")
    training_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(
        param_dict, training_dataset, validation_dataset, testing_dataset, param_dict['split_strategy'])

    # Model Construction
        # 为了避免过多的随机性影响，尽量保证在同一个初始的模型开始训练
    global_init_model_dir = r"./save_path/Ablation/" + param_dict['ablation_name'] + "/" + param_dict['dataset']
    check_and_make_the_path(global_init_model_dir)
    global_model = Experiment_Create_model(param_dict)

    if "SENT_CLF" in param_dict["task"]:
        global_init_model_path = global_init_model_dir + "/global_model_init.pt"
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            global_model.load_state_dict(torch.load(global_init_model_path, weights_only=False).state_dict())
        global_model.bert.finetune = True
        global_model.out.finetune = True
    elif "IMG_CLF" in param_dict["task"]:
        global_init_model_path = global_init_model_dir + "/global_model_4_IMG_CLF_init.pt"
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            global_model.load_state_dict(torch.load(global_init_model_path, weights_only=False).state_dict())
    elif "Tabular_CLF" in param_dict["task"]:
        global_init_model_path = global_init_model_dir + "/global_model_4_Tabular_CLF_init.pt"
        if not os.path.exists(global_init_model_path):
            torch.save(global_model, global_init_model_path)
        else:
            global_model.load_state_dict(torch.load(global_init_model_path, weights_only=False).state_dict())


    # PDFFed
    logger.info("~~~~~~ Algorithm: PDFFed ~~~~~~")
    Experiment_FL(eval(param_dict['ablation_name']), param_dict, global_model, training_dataloaders, training_dataset,
                  client_dataset_list, testing_dataloader, testing_dataset)


