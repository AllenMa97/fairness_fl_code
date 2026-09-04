import gc
import inspect
import os
import sys
import torch
import copy
import numpy as np
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable
from functools import partial

from tool.logger import *
from tool.utils import check_and_make_the_path, FL_fairness_and_accuracy_test, FL_fairness_and_accuracy_test_4_IMG_CLF, FL_fairness_and_accuracy_test_4_Tabular_CLF, get_HM_by_two_value
from tool.checkpoint import (
    CheckpointCompatibilityError,
    clear_repeat_artifacts,
    finalize_repeat_artifacts,
    load_checkpoint,
    load_repeat_metrics,
    restore_rng_state,
    save_aggregate_metrics,
    save_checkpoint,
    save_repeat_metrics,
)
from tool.experiment_state import (
    AlgorithmRunResult,
    RepeatResult,
    aggregate_repeat_results,
    capture_resource_snapshot,
    normalize_algorithm_result,
)
from tool.seed_manager import get_repeat_seed, set_all_seeds
from tool.checkpoint import build_experiment_config_hash
from module.experiment_setup import Experiment_Create_dataset, Experiment_Create_dataloader, Experiment_Create_model
from tool.tensorboard_logger import init_tensorboard_logger, log_test_metrics, log_system_metrics, flush, close
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
from algorithm.FederatedSum import Fed_Sum
from algorithm.FederatedAverageWithPo import Fed_AVG_Po
from algorithm.DOSFL import DistilledOneShotFed
# from algorithm.abandon.PoTrain import PoTrain
from algorithm.NaiveMix import NaiveMix
from algorithm.FedMix import FedMix
from algorithm.mFairFL import mFairFL
from algorithm.PDFFed import PDF_Fed
from algorithm.PDFFed_DP import PDF_Fed_DP
from algorithm.PraFFL import PraFFL
from tool.praffl_evaluation import evaluate_praffl
from algorithm.FedFACT import FedFACT
from algorithm.fedfact_evaluation import evaluate_fedfact
from algorithm.LoGoFair import LoGoFair
from algorithm.ProxProbability import ProxProbability
from algorithm.backup.DENSE import DENSE
from algorithm.backup.FENS import FENS
from algorithm.backup.FedCAV import FedCAV
from algorithm.backup.FedDEO import FedDEO
from algorithm.backup.FedELMY import FedELMY
from algorithm.backup.FedFisher import FedFisher
from algorithm.backup.FedKD import FedKD
# ===== 新增 11 个算法（KD / 梯度 / 特征 / 输入空间系列）=====
from algorithm.FedLGD import Fed_LGD
from algorithm.FedGen import Fed_Gen
from algorithm.FedDF import Fed_DF
from algorithm.FedET import Fed_ET
from algorithm.FedOMG import Fed_OMG
from algorithm.MAHyFL import MA_HyFL
from algorithm.FedFed import Fed_Fed
from algorithm.FedFree import Fed_Free
from algorithm.FedF2DG import Fed_F2DG
from algorithm.FedCOG import Fed_COG
from algorithm.FedRevive import Fed_Revive
# ===== 新增 9 个算法（One-shot / Bayesian / Analytic / 生成式系列）=====
from algorithm.FedBE import Fed_BE
from algorithm.FedCVAE import Fed_CVAE_Ens, Fed_CVAE_KD
from algorithm.FedBEns import Fed_BEns
from algorithm.FAFI import FAFI
from algorithm.FedTMOS import Fed_TMOS
from algorithm.FOL import FOL
from algorithm.FedLMG import Fed_LMG
from algorithm.AFL import AFL
from algorithm.GeFL import GeFL, GeFL_F
from algorithm.FairFedMOE import Fair_FedMOE
from ablation.PDFFed_Abl import *
from ablation.PDFFed_V2_Abl import *



@dataclass(frozen=True)
class AlgorithmRegistration:
    algorithm_function: Callable
    evaluator_function: Callable | None = None


def get_fedfact_registration(name):
    if name != "FedFACT":
        raise ValueError(
            f"Unknown algorithm: {name}; only paper-faithful FedFACT-In is registered"
        )
    return AlgorithmRegistration(FedFACT, evaluate_fedfact)

def calculate_communication_cost(algorithm_name, param_dict, global_model):
    I = param_dict['communication_round_I']
    K = param_dict['num_clients_K']
    fraction = param_dict['FL_fraction']
    task = param_dict.get('task', '')

    model_MB = sum(p.numel() for p in global_model.parameters()) * 4 / (1024 * 1024)

    if "SENT_CLF" in task:
        emb_dim = param_dict.get('emb_dim', 768)
        rep_MB = (
            sum(p.numel() for p in global_model.bert.parameters()) * 4 / (1024 * 1024)
            if hasattr(global_model, "bert") else model_MB
        )
        clf_params_count = (
            sum(p.numel() for p in global_model.out.parameters())
            if hasattr(global_model, "out") else 0
        )
    elif "IMG_CLF" in task:
        emb_dim = param_dict.get('emb_dim', 512)
        rep_MB = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
        clf_params_count = sum(p.numel() for p in global_model.out_layer.parameters())
    elif "Tabular_CLF" in task:
        emb_dim = param_dict.get('emb_dim', param_dict.get('nn_input_size', 128))
        if hasattr(global_model, 'shared_base'):
            rep_MB = sum(p.numel() for p in global_model.shared_base.parameters()) * 4 / (1024 * 1024)
            clf_params_count = sum(p.numel() for p in global_model.out_layer.parameters())
        else:
            # LogisticRegression 等无 shared_base 的模型：整个模型即为 rep
            rep_MB = model_MB
            clf_params_count = 0
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

    # ---- FedSum: 上传模型+语义画像(2类原型), 下载模型+分发预测头 ----
    elif algorithm_name == "Fed_Sum":
        clf_MB = clf_params_count * 4 / (1024 * 1024)
        cost = I * selected_per_round * (2 * model_MB + prototype_MB + clf_MB)

    # ---- Fed_AVG_Po: 上传模型+2类原型, 下载模型 ----
    elif algorithm_name == "Fed_AVG_Po":
        cost = I * selected_per_round * (2 * model_MB + prototype_MB)

    # ---- PDFFed: 上传model+4组群组原型, 下载model ----
    elif algorithm_name == "PDF_Fed" or algorithm_name == "PDF_Fed_DP":
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

    # ---- PraFFL: only the communicated BERT encoder is uploaded/downloaded ----
    elif algorithm_name == "PraFFL":
        if task != "SENT_CLF" or not hasattr(global_model, "bert"):
            raise ValueError(
                "PraFFL communication accounting requires a SENT_CLF BERT encoder"
            )
        selected_count = max(int(fraction * K), 1)
        if float(param_dict.get("FL_drop_rate", 0.0)) != 0.0:
            selected_count -= max(
                int(selected_count * float(param_dict["FL_drop_rate"])),
                1,
            )
        if selected_count < 1:
            raise ValueError("PraFFL FL_drop_rate leaves no selected clients")
        encoder_mb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in global_model.bert.state_dict().values()
        ) / (1024 * 1024)
        cost = I * selected_count * 2 * encoder_mb

    # ---- FedFACT: unified download + unified update upload; personal model stays private ----
    elif algorithm_name == "FedFACT":
        cost = I * K * 2 * model_MB

    # ---- FedLGD: 标准模型通信 + 梯度匹配（梯度向量大小≈model大小，近似2x ----
    elif algorithm_name == "Fed_LGD":
        cost = I * selected_per_round * 2 * model_MB * 1.1

    # ---- FedGen: 下载model, 上传model + 轻量生成器 (生成器 < model, 加10%余量) ----
    elif algorithm_name == "Fed_Gen":
        cost = I * selected_per_round * 2 * model_MB * 1.15

    # ---- FedDF / Fed-ET: 标准模型通信 + ensemble logit KD（logits 近似忽略） ----
    elif algorithm_name in ["Fed_DF", "Fed_ET"]:
        cost = I * selected_per_round * 2 * model_MB

    # ---- FedOMG: 标准模型 + 上传梯度向量（~model），近似 1.1x ----
    elif algorithm_name == "Fed_OMG":
        cost = I * selected_per_round * 2 * model_MB * 1.1

    # ---- MA-HyFL: 标准模型 + 双向 logits（近似忽略） ----
    elif algorithm_name == "MA_HyFL":
        cost = I * selected_per_round * 2 * model_MB

    # ---- FedFed: 标准模型通信 + 上传特征统计量(均值/方差，维度~emb_dim，近似 1.05x) ----
    elif algorithm_name == "Fed_Fed":
        cost = I * selected_per_round * 2 * model_MB * 1.05

    # ---- FedFree: 标准模型 + 逐层 center（层数*emb_dim，近似 1.1x） ----
    elif algorithm_name == "Fed_Free":
        cost = I * selected_per_round * 2 * model_MB * 1.1

    # ---- FedF²DG: 标准模型 + 伪输入向量（样本数*feat_dim，近似 1.1x） ----
    elif algorithm_name == "Fed_F2DG":
        cost = I * selected_per_round * 2 * model_MB * 1.1

    # ---- FedCOG: 标准模型 + 共识生成器 + 共识特征（近似 1.15x） ----
    elif algorithm_name == "Fed_COG":
        cost = I * selected_per_round * 2 * model_MB * 1.15

    # ---- FedRevive: 标准模型 + Meta Generator + stale buffer（近似 1.2x） ----
    elif algorithm_name == "Fed_Revive":
        cost = I * selected_per_round * 2 * model_MB * 1.2

    # ---- FedBE: 标准模型通信（medoid+集成蒸馏在服务器端完成） ----
    elif algorithm_name == "Fed_BE":
        cost = I * selected_per_round * 2 * model_MB

    # ---- FedCVAE-Ens/KD: 标准模型 + 上传 CVAE 解码器/label_emb/分类头（~0.2x） ----
    elif algorithm_name in ["Fed_CVAE_Ens", "Fed_CVAE_KD"]:
        cost = I * selected_per_round * 2 * model_MB * 1.2

    # ---- FedBEns: 标准模型 + 对角经验 Fisher（≈1x model）= 3x ----
    elif algorithm_name == "Fed_BEns":
        cost = I * selected_per_round * 3 * model_MB

    # ---- FAFI: 标准模型通信（α 插值与蒸馏在服务器端完成） ----
    elif algorithm_name == "FAFI":
        cost = I * selected_per_round * 2 * model_MB

    # ---- FedTMOS: 标准模型通信（自动机投票在服务器端完成） ----
    elif algorithm_name == "Fed_TMOS":
        cost = I * selected_per_round * 2 * model_MB

    # ---- FOL: 标准模型 + 逐类特征中心与分类头（近似 1.1x） ----
    elif algorithm_name == "FOL":
        cost = I * selected_per_round * 2 * model_MB * 1.1

    # ---- FedLMG: 标准模型 + 分类头上传（近似 1.05x） ----
    elif algorithm_name == "Fed_LMG":
        cost = I * selected_per_round * 2 * model_MB * 1.05

    # ---- AFL: 闭式解，上传 A/B 统计量（(d+1)^2 + (d+1)*c，远小于 model）----
    elif algorithm_name == "AFL":
        cost = I * selected_per_round * 1.1 * model_MB

    # ---- GeFL / GeFL-F: 标准模型 + 分类头 + 类中心（近似 1.1x） ----
    elif algorithm_name in ["GeFL", "GeFL_F"]:
        cost = I * selected_per_round * 2 * model_MB * 1.1

    # ---- Fair_FedMOE: 标准模型（含 MoE 原型/专家头，远小于骨干，近似 1.05x） ----
    elif algorithm_name == "Fair_FedMOE":
        cost = I * selected_per_round * 2 * model_MB * 1.05

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
        if "SENT_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test(global_model, param_dict, testing_dataloader, testing_dataset_len)
        elif "IMG_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_IMG_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
        elif "Tabular_CLF" in param_dict["task"]:
            accuracy, DEO, SPD = FL_fairness_and_accuracy_test_4_Tabular_CLF(global_model, param_dict, testing_dataloader, testing_dataset_len)
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


def _evaluate_global_model(global_model, param_dict, data_bundle, algorithm_state):
    """Default final evaluator; algorithm-specific baselines can supply a hook."""
    del algorithm_state
    loader = data_bundle.testing_dataloader
    testing_dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else len(loader)
    if "SENT_CLF" in param_dict["task"]:
        accuracy, deo, spd = FL_fairness_and_accuracy_test(
            global_model, param_dict, loader, testing_dataset_len
        )
    elif "IMG_CLF" in param_dict["task"]:
        accuracy, deo, spd = FL_fairness_and_accuracy_test_4_IMG_CLF(
            global_model, param_dict, loader, testing_dataset_len
        )
    elif "Tabular_CLF" in param_dict["task"]:
        accuracy, deo, spd = FL_fairness_and_accuracy_test_4_Tabular_CLF(
            global_model, param_dict, loader, testing_dataset_len
        )
    else:
        raise ValueError(f"unsupported evaluation task: {param_dict['task']}")
    metrics = {"ACC": float(accuracy), "DEO": float(deo), "SPD": float(spd)}
    metrics["FR"] = 1.0 - metrics["DEO"]
    metrics["HM"] = float(get_HM_by_two_value(metrics["ACC"], metrics["FR"]))
    return metrics


def _algorithm_accepts_resume_state(algorithm_function):
    signature = inspect.signature(algorithm_function)
    return "resume_state" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _algorithm_accepts_data_bundle(algorithm_function):
    signature = inspect.signature(algorithm_function)
    return "data_bundle" in signature.parameters


def _log_final_evaluation_to_tensorboard(metrics, param_dict, run_result):
    """The runner owns exactly one terminal TensorBoard evaluation event."""
    required = {"ACC", "DEO", "SPD"}
    if not required.issubset(metrics):
        return
    kwargs = {
        "accuracy": float(metrics["ACC"]),
        "DEO": float(metrics["DEO"]),
        "SPD": float(metrics["SPD"]),
        "step": int(param_dict["communication_round_I"]),
        "gpu_seconds": float(run_result.total_gpu_seconds),
        "communication_cost": float(run_result.total_communication_cost),
        "prefix": "final/",
    }
    if "FR" in metrics:
        kwargs["FR"] = float(metrics["FR"])
    if "HM" in metrics:
        kwargs["HM"] = float(metrics["HM"])
    log_test_metrics(**kwargs)
    flush()


def _run_single_repeat(repeat_idx, algorithm_function, evaluator_function, param_dict):
    """Run one self-contained repeat, constructing all randomized inputs after seeding."""
    repeat_param = dict(param_dict)
    repeat_seed = get_repeat_seed(repeat_idx=repeat_idx, base_seed=int(param_dict.get("base_seed", 42)))
    repeat_param.update({"repeat_idx": int(repeat_idx), "repeat_seed": int(repeat_seed)})
    set_all_seeds(repeat_seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Construction must be after set_all_seeds: paired repeat r therefore has the
    # same data split, loader order and initialization for every algorithm.
    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(repeat_param)
    data_bundle = Experiment_Create_dataloader(
        repeat_param, training_dataset, validation_dataset, testing_dataset,
        repeat_param["split_strategy"],
    )
    if not hasattr(data_bundle, "partition_fingerprint"):
        raise TypeError("Experiment_Create_dataloader must return a FederatedDataBundle")
    repeat_param["partition_fingerprint"] = data_bundle.partition_fingerprint
    repeat_param["partition_metadata"] = data_bundle.partition_metadata
    config_hash = str(param_dict.get("experiment_config_hash") or build_experiment_config_hash(param_dict))
    repeat_param["experiment_config_hash"] = config_hash

    if repeat_param.get("resume", False):
        completed = load_repeat_metrics(
            repeat_param, repeat_idx, config_hash, data_bundle.partition_fingerprint
        )
        if completed is not None:
            return RepeatResult(
                repeat_idx, repeat_seed, data_bundle.partition_fingerprint,
                completed["metrics"], float(completed["total_gpu_seconds"]),
                float(completed["total_communication_cost"]),
                resource_usage=dict(completed["resource_usage"]),
            )
    else:
        # Fresh is explicit: stale checkpoints/metrics never affect a new run.
        clear_repeat_artifacts(repeat_param, repeat_idx)

    global_model = Experiment_Create_model(repeat_param)
    resume_state = None
    if repeat_param.get("resume", False):
        resume_state = load_checkpoint(
            repeat_param,
            expected_config_hash=config_hash,
            expected_partition_fingerprint=data_bundle.partition_fingerprint,
            expected_repeat_idx=repeat_idx,
        )
    if resume_state is not None:
        global_model.load_state_dict(resume_state.global_model_state)
        # Dataset/loaders/model construction may consume RNG. Restore at the saved
        # round boundary only after they are rebuilt.
        restore_rng_state(resume_state)

    wall_start = time.monotonic()
    if resume_state is not None and resume_state.phase == "evaluate":
        # A final-round checkpoint is deliberately not completion.  It permits a
        # crashed final evaluator to resume without retraining.
        run_result = AlgorithmRunResult(
            global_model=global_model,
            total_gpu_seconds=resume_state.total_gpu_seconds,
            total_communication_cost=resume_state.total_communication_cost,
            algorithm_state=resume_state.algorithm_state,
            amp_scaler_state=resume_state.amp_scaler_state,
            client_selection_history=resume_state.client_selection_history,
        )
        prior_runtime = resume_state.total_runtime_seconds
    else:
        accepts_resume_state = _algorithm_accepts_resume_state(algorithm_function)
        if resume_state is not None and not accepts_resume_state:
            if (resume_state.algorithm_state or resume_state.amp_scaler_state or
                    resume_state.total_gpu_seconds or resume_state.total_communication_cost or
                    resume_state.client_selection_history):
                raise CheckpointCompatibilityError(
                    f"{algorithm_function.__name__} cannot restore algorithm/AMP/counter state"
                )
        kwargs = {"start_round": 0 if resume_state is None else resume_state.next_round}
        if accepts_resume_state:
            kwargs["resume_state"] = resume_state
        if _algorithm_accepts_data_bundle(algorithm_function):
            kwargs["data_bundle"] = data_bundle
        raw_result = algorithm_function(
            repeat_param["device"], global_model,
            repeat_param["algorithm_epoch_T"], repeat_param["num_clients_K"],
            repeat_param["communication_round_I"], repeat_param["FL_fraction"],
            repeat_param["FL_drop_rate"], data_bundle.training_dataloaders,
            training_dataset, data_bundle.client_dataset_list, repeat_param,
            data_bundle.testing_dataloader, len(testing_dataset), **kwargs,
        )
        run_result = normalize_algorithm_result(raw_result)
        prior_runtime = 0.0 if resume_state is None else resume_state.total_runtime_seconds

    # Save an explicit terminal train boundary before final evaluation.  Only the
    # subsequently written metrics.json marks the repeat as complete.
    checkpoint_path = save_checkpoint(
        repeat_param, int(repeat_param["communication_round_I"]) - 1,
        run_result.global_model, algorithm_state=run_result.algorithm_state,
        amp_scaler=run_result.amp_scaler_state,
        total_gpu_seconds=run_result.total_gpu_seconds,
        total_runtime_seconds=prior_runtime + (time.monotonic() - wall_start),
        total_communication_cost=run_result.total_communication_cost,
        client_selection_history=run_result.client_selection_history,
    )
    resource_usage = capture_resource_snapshot(checkpoint_path)
    evaluator = evaluator_function or _evaluate_global_model
    metrics = evaluator(run_result.global_model, repeat_param, data_bundle, run_result.algorithm_state)
    _log_final_evaluation_to_tensorboard(metrics, repeat_param, run_result)
    save_repeat_metrics(
        repeat_param, repeat_idx, config_hash, data_bundle.partition_fingerprint, metrics,
        repeat_seed=repeat_seed, total_gpu_seconds=run_result.total_gpu_seconds,
        total_communication_cost=run_result.total_communication_cost,
        resource_usage=resource_usage,
    )
    finalize_repeat_artifacts(
        repeat_param, repeat_idx, run_result.global_model,
        repeat_param.get("final_artifact_policy", "metrics_only"),
    )
    return RepeatResult(
        repeat_idx, repeat_seed, data_bundle.partition_fingerprint, metrics,
        run_result.total_gpu_seconds, run_result.total_communication_cost,
        resource_usage=resource_usage,
    )


def _append_human_readable_aggregate(param_dict, algorithm_name, aggregate):
    with open(param_dict["result_path"], "a+", encoding="utf-8") as stream:
        for name, stats in aggregate["metrics"].items():
            stream.write(f"{algorithm_name} {name} Mean±STD: {stats['mean']:.3f}±{stats['std']:.3f}\n")
        stream.write("----------------------------------------------------------------------------\n")


def Experiment_FL(algorithm_function, param_dict, evaluator_function=None):
    """Schedule repeats only; every path calls the same single-repeat worker."""
    if not param_dict.get("redraw_partition_per_repeat", False):
        # Default (fixed partition): leave the flag out of the experiment-config
        # hash so fixed-partition runs share one stable identity and resume cleanly.
        param_dict.pop("redraw_partition_per_repeat", None)
    repeats = int(param_dict.get("exp_repeat_times", 3))
    parallel = max(1, min(int(param_dict.get("parallel_repeats", 1)), repeats))
    if str(param_dict.get("device", "cpu")).startswith("cuda") and parallel != 1:
        raise ValueError("CUDA repeats must run serially; set parallel_repeats=1")
    config_hash = build_experiment_config_hash(param_dict)
    run_param = dict(param_dict, experiment_config_hash=config_hash)
    args = [(index, algorithm_function, evaluator_function, run_param) for index in range(repeats)]
    if parallel > 1:
        context = mp.get_context("spawn")
        with context.Pool(processes=parallel) as pool:
            results = pool.starmap(_run_single_repeat, args)
    else:
        results = [_run_single_repeat(*arguments) for arguments in args]
    aggregate = aggregate_repeat_results(results, expected_repeats=repeats)
    save_aggregate_metrics(run_param, aggregate)
    _append_human_readable_aggregate(run_param, algorithm_function.__name__, aggregate)
    return aggregate

def Experiment_pFL(algorithm_function, param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader, testing_dataset):
    pass


def _create_legacy_single_run_inputs(param_dict):
    # # Create dataset
    logger.info("Creating dataset")
    training_dataset, validation_dataset, testing_dataset = Experiment_Create_dataset(param_dict)

    # Create dataloader
    logger.info("Creating dataloader")
    data_bundle = Experiment_Create_dataloader(
        dict(param_dict, repeat_idx=0), training_dataset, validation_dataset, testing_dataset,
        param_dict['split_strategy'])
    training_dataloaders = data_bundle.training_dataloaders
    client_dataset_list = data_bundle.client_dataset_list
    testing_dataloader = data_bundle.testing_dataloader

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

    return training_dataset, testing_dataset, data_bundle, global_model


def _run_praffl_experiment(param_dict):
    return Experiment_FL(
        PraFFL,
        param_dict,
        evaluator_function=evaluate_praffl,
    )


def Experiment(param_dict):
    # 统一 AMP 控制：根据 GPU 能力自动决定是否启用混合精度
    from tool.amp_utils import resolve_amp_config
    param_dict['use_amp'] = resolve_amp_config(param_dict)

    # 添加 client_parallel 参数支持（如果未设置则默认为 'auto'）
    if 'client_parallel' not in param_dict:
        param_dict['client_parallel'] = 'auto'

    # 初始化TensorBoard日志记录器
    try:
        tb_logger = init_tensorboard_logger(
            experiment_name=param_dict.get('Experiment_NO', 'exp'),
            algorithm=param_dict.get('algorithm', 'unknown'),
            dataset=param_dict.get('dataset_name', param_dict.get('dataset', 'unknown')),
            split_strategy=param_dict.get('split_strategy'),
            hypothesis=param_dict.get('hypothesis'),
            num_clients_K=param_dict.get('num_clients_K'),
            base_log_dir=param_dict.get('tb_log_dir')
        )
        logger.info("TensorBoard logging initialized")
        try:
            from tool.tensorboard_logger import log_experiment_config, get_monitoring_config
            # 记录实验配置到TensorBoard
            safe_config = {k: v for k, v in param_dict.items()
                          if isinstance(v, (str, int, float, bool, list, tuple))}
            log_experiment_config(safe_config)

            # 加载 TensorBoard 监控配置（全部默认开启）
            tb_cfg = get_monitoring_config(param_dict)
            logger.info(f"[TensorBoard] All monitoring modules active; to disable any, set param_dict['tb_monitor']")
        except Exception:
            pass
    except ImportError:
        logger.warning("TensorBoard not installed, skipping TensorBoard logging")
        tb_logger = None
    except Exception as e:
        logger.warning(f"Failed to initialize TensorBoard logging: {e}")
        tb_logger = None

    # SeparateTraining
    if ("Separate" in param_dict["algorithm"]) or ("separate" in param_dict["algorithm"]) or (
            "sepa" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: SeparateTraining ~~~~~~")
        training_dataset, testing_dataset, data_bundle, global_model = _create_legacy_single_run_inputs(param_dict)
        Experiment_SeparateTraining(
            param_dict, global_model, data_bundle.training_dataloaders, training_dataset,
            data_bundle.client_dataset_list, data_bundle.testing_dataloader, testing_dataset
        )
    # CentralizedTraining
    elif ("Centralized" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: CentralizedTraining ~~~~~~")
        training_dataset, testing_dataset, data_bundle, global_model = _create_legacy_single_run_inputs(param_dict)
        Experiment_SeparateTraining(
            param_dict, global_model, data_bundle.training_dataloaders, training_dataset,
            data_bundle.client_dataset_list, data_bundle.testing_dataloader, testing_dataset

    )
    # Federated Average
    elif ("FederatedAverage" in param_dict["algorithm"]) or ("FedAvg" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Average ~~~~~~")
        Experiment_FL(Fed_AVG, param_dict)
    # Federated Prox
    elif ("FederatedProximal" in param_dict["algorithm"]) or ("FedProx" in param_dict["algorithm"]) or (
            "fedprox" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Proximal ~~~~~~")
        Experiment_FL(Fed_Prox, param_dict)


    # SCAFFOLD
    elif ("Scaffold" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Scaffold ~~~~~~")
        Experiment_FL(Scaffold, param_dict)


    # Federated Nova
    elif ("FederatedNova" in param_dict["algorithm"]) or ("FedNova" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Nova ~~~~~~")
        Experiment_FL(Fed_Nova, param_dict)

    # FedRep
    elif ("FedRep" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Rep ~~~~~~")
        Experiment_FL(Fed_Rep, param_dict)

    # FedProto
    elif ("FedProto" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Federated Proto ~~~~~~")
        Experiment_FL(Fed_PROTO, param_dict)

    # One-Shot Federated Learning
    elif ("OSFL" in param_dict["algorithm"]) and (param_dict["algorithm"] != "DOSFL"):
        logger.info("~~~~~~ Algorithm: One-Shot Federated Learning ~~~~~~")
        Experiment_FL(OneShotFed, param_dict)

    # CO_BOOSTING
    elif ("CO_BOOSTING" in param_dict["algorithm"]) or ("CoBoosting" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: CO-BOOSTING ~~~~~~")
        Experiment_FL(Co_Boosting, param_dict)

    # DOSFL
    elif ("DistilledOneShotFed" in param_dict["algorithm"]) or ("DOSFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: DOSFL ~~~~~~")
        Experiment_FL(DistilledOneShotFed, param_dict)

    # FedFair
    elif ("FedFair" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFair ~~~~~~")
        Experiment_FL(FedFair, param_dict)

    # FL_FairBatch
    elif ("FL_FairBatch" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FL_FairBatch ~~~~~~")
        Experiment_FL(FL_FairBatch, param_dict)

    # FedFB
    elif ("FedFB" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFB ~~~~~~")
        Experiment_FL(FedFB, param_dict)

    # FairFed
    elif ("FairFed" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FairFed ~~~~~~")
        Experiment_FL(FairFed, param_dict)

    # FedRenyi
    elif ("FedRenyi" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedRenyi ~~~~~~")
        Experiment_FL(Fed_Renyi, param_dict)

    # FedSum
    elif ("FedSum" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedSum ~~~~~~")
        Experiment_FL(Fed_Sum, param_dict)

    # Fed_AVG_Po (FederatedAverageWithPo: FedAvg + Class Prototype + L_Po)
    elif ("FederatedAverageWithPo" in param_dict["algorithm"]) or ("Fed_AVG_Po" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Fed_AVG_Po (FedAvg + Class Prototype + L_Po) ~~~~~~")
        Experiment_FL(Fed_AVG_Po, param_dict, global_model, training_dataloaders, training_dataset,
                      client_dataset_list, testing_dataloader, testing_dataset)

    # FedMix
    elif ("FedMix" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedMix ~~~~~~")
        Experiment_FL(FedMix, param_dict)

    # NaiveMix
    elif ("NaiveMix" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: NaiveMix ~~~~~~")
        Experiment_FL(NaiveMix, param_dict)

    # mFairFL
    elif ("mFairFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: mFairFL ~~~~~~")
        Experiment_FL(mFairFL, param_dict)

    # PDFFed_DP: PDFFed with Differential Privacy（对应挑战1/定理5）
    elif ("PDFFed_DP" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: PDFFed_DP (PDFFed + LDP noise) ~~~~~~")
        Experiment_FL(PDF_Fed_DP, param_dict)

    # PDFFed
    elif ("PDFFed" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: PDFFed ~~~~~~")
        Experiment_FL(PDF_Fed, param_dict)

    # PraFFL (KDD 2025)
    elif ("PraFFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: PraFFL ~~~~~~")
        _run_praffl_experiment(param_dict)

    # FedFACT-In (NeurIPS 2025)
    elif str(param_dict["algorithm"]).startswith("FedFACT"):
        logger.info("~~~~~~ Algorithm: FedFACT-In ~~~~~~")
        registration = get_fedfact_registration(param_dict["algorithm"])
        Experiment_FL(
            registration.algorithm_function,
            param_dict,
            evaluator_function=registration.evaluator_function,
        )

    # LoGoFair
    elif ("LoGoFair" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: LoGoFair ~~~~~~")
        Experiment_FL(LoGoFair, param_dict)

    # DENSE (NeurIPS 2022)
    elif ("DENSE" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: DENSE ~~~~~~")
        Experiment_FL(DENSE, param_dict)

    # FENS (NeurIPS 2024)
    elif ("FENS" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FENS ~~~~~~")
        Experiment_FL(FENS, param_dict)

    # FedCAV (ICLR 2023)
    elif ("FedCAV" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedCAV ~~~~~~")
        Experiment_FL(FedCAV, param_dict)

    # FedDEO (ACM MM 2024)
    elif ("FedDEO" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedDEO ~~~~~~")
        Experiment_FL(FedDEO, param_dict)

    # FedELMY (ACM MM 2024)
    elif ("FedELMY" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedELMY ~~~~~~")
        Experiment_FL(FedELMY, param_dict)

    # FedFisher (AISTATS 2024)
    elif ("FedFisher" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFisher ~~~~~~")
        Experiment_FL(FedFisher, param_dict)

    # FedKD (AAAI 2022)
    elif ("FedKD" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedKD ~~~~~~")
        Experiment_FL(FedKD, param_dict)

    # ProxProbability
    elif ("ProxProbability" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: ProxProbability ~~~~~~")
        Experiment_FL(ProxProbability, param_dict)

    # ========== 新增 11 个算法入口 ==========

    # FedLGD (arXiv 2023, Gradient Matching)
    elif ("FedLGD" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedLGD (Local-Global Distillation / Gradient Space) ~~~~~~")
        Experiment_FL(Fed_LGD, param_dict)

    # FedGen (ICML 2021, Latent Space Generator)
    elif ("FedGen" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedGen (Data-Free KD / Latent Generator) ~~~~~~")
        Experiment_FL(Fed_Gen, param_dict)

    # FedDF (NeurIPS 2020, Ensemble Logit KD + Proxy Data)
    elif ("FedDF" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedDF (Ensemble Distillation + Proxy Data) ~~~~~~")
        Experiment_FL(Fed_DF, param_dict)

    # Fed-ET (IJCAI 2022, Weighted Consensus Distillation + Diversity)
    elif ("FedET" in param_dict["algorithm"]) or ("Fed-ET" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Fed-ET (Heterogeneous Ensemble Knowledge Transfer) ~~~~~~")
        Experiment_FL(Fed_ET, param_dict)

    # FedOMG (ICLR 2025, Gradient Inner Product Maximization)
    elif ("FedOMG" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedOMG (On-server Matching Gradient) ~~~~~~")
        Experiment_FL(Fed_OMG, param_dict)

    # MA-HyFL (TCSVT 2026, Bidirectional Cross-Modal KD + RL Aggregation)
    elif ("MAHyFL" in param_dict["algorithm"]) or ("MA-HyFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: MA-HyFL (Modality-Agnostic Hybrid FL) ~~~~~~")
        Experiment_FL(MA_HyFL, param_dict)

    # FedFed (NeurIPS 2023, Feature / Activation Distillation Alignment)
    elif ("FedFed" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFed (Feature Distillation) ~~~~~~")
        Experiment_FL(Fed_Fed, param_dict)

    # FedFree (NeurIPS 2025, Layer-wise Activation + KGE)
    elif ("FedFree" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedFree (Layer-wise Alignment + KGE) ~~~~~~")
        Experiment_FL(Fed_Free, param_dict)

    # FedF²DG (Neural Networks 2024, Generator-free Model Inversion Pseudo-input)
    elif ("FedF2DG" in param_dict["algorithm"]) or ("FedF" in param_dict["algorithm"]) or ("FedFSq" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedF²DG (Generator-free Data Generation) ~~~~~~")
        Experiment_FL(Fed_F2DG, param_dict)

    # FedCOG (ICLR 2024, Consensus-Oriented Generation)
    elif ("FedCOG" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedCOG (Consensus-Oriented Generation) ~~~~~~")
        Experiment_FL(Fed_COG, param_dict)

    # FedRevive (arXiv 2025, Meta-Generator + Stale Update Revival)
    elif ("FedRevive" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedRevive (Stale Update Revival + Meta-Generator) ~~~~~~")
        Experiment_FL(Fed_Revive, param_dict)

    # ========== 新增 9 个算法入口（One-shot / Bayesian / Analytic / 生成式系列）==========
    # 注意：子串匹配按从特殊到一般的顺序排列（如 FedBEns 在 FedBE 之前、GeFL_F 在 GeFL 之前）

    # Fair-FedMOE (ICML 2026, Prototype-Guided Experts for Group-Fair OFL)
    elif ("Fair-FedMOE" in param_dict["algorithm"]) or ("Fair_FedMOE" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: Fair-FedMOE (Group-Fair FL via Prototype-Guided Experts) ~~~~~~")
        Experiment_FL(Fair_FedMOE, param_dict)

    # FedBEns (ICML 2025, Laplace-approximated Bayesian Ensemble)
    elif ("FedBEns" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedBEns (One-Shot FL via Bayesian Ensemble) ~~~~~~")
        Experiment_FL(Fed_BEns, param_dict)

    # FedBE (ICLR 2021, Bayesian Model Ensemble)
    elif ("FedBE" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedBE (Bayesian Model Ensemble Applicable to FL) ~~~~~~")
        Experiment_FL(Fed_BE, param_dict)

    # FedCVAE-KD / FedCVAE-Ens (ICLR 2026, CVAE-based Data-Free One-Shot FL)
    elif ("FedCVAE-KD" in param_dict["algorithm"]) or ("FedCVAE_KD" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedCVAE-KD (CVAE Reconstruction + Soft KD) ~~~~~~")
        Experiment_FL(Fed_CVAE_KD, param_dict)
    elif ("FedCVAE-Ens" in param_dict["algorithm"]) or ("FedCVAE_Ens" in param_dict["algorithm"]) or ("FedCVAE" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedCVAE-Ens (CVAE Reconstruction + Ensemble Vote) ~~~~~~")
        Experiment_FL(Fed_CVAE_Ens, param_dict)

    # FAFI (ICML 2025, Mitigating Model Inconsistency)
    elif ("FAFI" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FAFI (Mitigating Model Inconsistency in One-shot FL) ~~~~~~")
        Experiment_FL(FAFI, param_dict)

    # FedTMOS (ICLR 2025, Tsetlin Machine One-Shot FL)
    elif ("FedTMOS" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedTMOS (One-Shot FL with Tsetlin Machine) ~~~~~~")
        Experiment_FL(Fed_TMOS, param_dict)

    # FOL (ICML 2025, Federated Oriented Learning)
    elif ("FOL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FOL (Federated Oriented Learning) ~~~~~~")
        Experiment_FL(FOL, param_dict)

    # FedLMG (ICML 2025, Local Model-Guided Diffusion)
    elif ("FedLMG" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: FedLMG (Local Model-Guided Diffusion Models) ~~~~~~")
        Experiment_FL(Fed_LMG, param_dict)

    # AFL (CVPR 2025, Analytic FL with Pre-trained Models)
    elif ("AFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: AFL (Single-Round Analytic Federated Learning) ~~~~~~")
        Experiment_FL(AFL, param_dict)

    # GeFL-F (IEEE TC 2024, Feature-level Generative FL)
    elif ("GeFL-F" in param_dict["algorithm"]) or ("GeFL_F" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: GeFL-F (Model-Agnostic FL with Generative Models, Feature) ~~~~~~")
        Experiment_FL(GeFL_F, param_dict)

    # GeFL (IEEE TC 2024, Input-space Generative FL)
    elif ("GeFL" in param_dict["algorithm"]):
        logger.info("~~~~~~ Algorithm: GeFL (Model-Agnostic FL with Generative Models) ~~~~~~")
        Experiment_FL(GeFL, param_dict)

    else:
        raise ValueError(f'''Wrong algorithm name:{param_dict['algorithm']} It should be in the following type:
            [Separate | FedAvg | FedProx | Scaffold | FederatedNova | FedRep | FedProto| OSFL | CO_BOOSTING | DOSFL |
             FedFair | FL_FairBatch | FedFB | FairFed | mFairFL | PDFFed | PDFFed_DP | PraFFL | FedFACT |
             FedRenyi | FedSum | NaiveMix | FedMix |
             DENSE | FENS | FedCAV | FedDEO | FedELMY | FedFisher | FedKD | ProxProbability |
             FedLGD | FedGen | FedDF | FedET / Fed-ET | FedOMG | MAHyFL / MA-HyFL |
             FedFed | FedFree | FedF2DG / FedF²DG | FedCOG | FedRevive |
             FedBE | FedCVAE-Ens / FedCVAE-KD | FedBEns | FAFI | FedTMOS | FOL | FedLMG | AFL |
             GeFL / GeFL-F] ''')

    # 关闭TensorBoard日志记录器
    try:
        if tb_logger is not None:
            close()
            logger.info("TensorBoard logging closed")
    except Exception as e:
        logger.warning(f"Failed to close TensorBoard logging: {e}")


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
    # Ablations use the same deterministic worker; retain their historical single repeat.
    Experiment_FL(eval(param_dict['ablation_name']), dict(param_dict, exp_repeat_times=1))
