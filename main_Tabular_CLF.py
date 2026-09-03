import os
import json
import argparse
import re
import numpy as np

from tool.logger import *
from tool.utils import check_and_make_the_path
from tool.experiment_cli import add_experiment_state_arguments
from experiment import Experiment


# 尝试导入tensorboard，如果没有安装则给出提示
try:
    from torch.utils.tensorboard import SummaryWriter
    print("[TensorBoard] Support is available.")
    print("[TensorBoard] After training, run: tensorboard --logdir=./tb_logs")
    print("[TensorBoard] Or view all experiments: tensorboard --logdir=./tb_logs --bind_all")
except ImportError:
    print("[TensorBoard] Not installed. Install with: pip install tensorboard")
    print("[TensorBoard] Continuing without TensorBoard support...")


def analyze_experiment_log(log_file):
    """手工日志检查辅助函数；不用于决定训练是否完成。"""
    if not os.path.exists(log_file):
        return 0, False, False
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        test_count = len(re.findall(r'Trained Global Model Testing', content))
        has_summary = 'Mean' in content and 'STD' in content
        has_training = len(re.findall(r'Communication Round: \d+', content)) > 0
        
        return test_count, has_summary, has_training
    except:
        return 0, False, False


def calculate_and_append_summary(log_file, algorithm):
    """手工日志汇总辅助函数；不用于决定训练是否完成。"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        acc_pattern = r'ACC:\s*([\d.]+),\s*DEO:\s*([\d.-]+),\s*SPD:\s*([\d.-]+),\s*FR:\s*([\d.]+),\s*HM:\s*([\d.]+)'
        matches = re.findall(acc_pattern, content)
        
        if len(matches) >= 3:
            last_three = matches[-3:]
            accs = [float(m[0]) for m in last_three]
            deos = [float(m[1]) for m in last_three]
            spds = [float(m[2]) for m in last_three]
            frs = [float(m[3]) for m in last_three]
            hms = [float(m[4]) for m in last_three]
            
            acc_mean, acc_std = np.mean(accs), np.std(accs)
            deo_mean, deo_std = np.mean(deos), np.std(deos)
            spd_mean, spd_std = np.mean(spds), np.std(spds)
            fr_mean, fr_std = np.mean(frs), np.std(frs)
            hm_mean, hm_std = np.mean(hms), np.std(hms)
            
            summary_lines = [
                f"****** {algorithm} ACC Mean±STD: {acc_mean:.3f}±{acc_std:.3f} ******",
                f"****** {algorithm} DEO Mean±STD: {deo_mean:.3f}±{deo_std:.3f} ******",
                f"****** {algorithm} SPD Mean±STD: {spd_mean:.3f}±{spd_std:.3f} ******",
                f"****** {algorithm} FR Mean±STD: {fr_mean:.3f}±{fr_std:.3f} ******",
                f"****** {algorithm} HM Mean±STD: {hm_mean:.3f}±{hm_std:.3f} ******",
            ]
            
            with open(log_file, 'a', encoding='utf-8') as f:
                for line in summary_lines:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO    : {line}\n")
            
            print(f"  [SUMMARY] Calculated and appended summary statistics")
            return True
        else:
            print(f"  [WARNING] Not enough test results found ({len(matches)}), cannot calculate summary")
            return False
    except Exception as e:
        print(f"  [ERROR] Failed to calculate summary: {e}")
        return False


def Argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'test'])
    # parser.add_argument("-algorithm", default='Centralized', type=str)
    # parser.add_argument("-algorithm", default='Separate', type=str)
    # parser.add_argument("-algorithm", default='FedAvg', type=str)
    # parser.add_argument("-algorithm", default='FedProx', type=str)
    # parser.add_argument("-algorithm", default='Scaffold', type=str)
    # parser.add_argument("-algorithm", default='FedNova', type=str)
    # parser.add_argument("-algorithm", default='FedRep', type=str)
    # parser.add_argument("-algorithm", default='FedProto', type=str)
    # parser.add_argument("-algorithm", default='FedFB', type=str)
    # parser.add_argument("-algorithm", default='FedRenyi', type=str)
    # parser.add_argument("-algorithm", default='OSFL', type=str)
    # parser.add_argument("-algorithm", default='CO_BOOSTING', type=str)
    # parser.add_argument("-algorithm", default='FairFed', type=str)
    # parser.add_argument("-algorithm", default='FedFair', type=str)
    # parser.add_argument("-algorithm", default='FL_FairBatch', type=str)
    parser.add_argument("-algorithm", default='PDFFed', type=str)

    # parser.add_argument("-algorithm", default='DOSFL', type=str)
    # parser.add_argument("-algorithm", default='FedMix', type=str)
    # parser.add_argument("-algorithm", default='NaiveMix', type=str)
    # parser.add_argument("-algorithm", default='ProxProbability', type=str)
    # parser.add_argument("-algorithm", default='Progressive', type=str)
    # parser.add_argument("-algorithm", default='ISOMOProgre', type=str)


    # parser.add_argument("-algorithm", default='GroupProto', type=str) # 废弃
    # parser.add_argument("-algorithm", default='GroupAlign', type=str) # 废弃
    # parser.add_argument("-algorithm", default='GroupAlignProto', type=str) # 废弃
    # parser.add_argument("-algorithm", default='PoTrain', type=str) # 废弃
    # parser.add_argument("-algorithm", default='GroupAlignProtoPoTrain', type=str) # 废弃
    # parser.add_argument("-algorithm", default='GroupDemographicAlign', type=str) # 废弃
    # parser.add_argument("-algorithm", default='AggregatedProgressive', type=str) # 废弃

    # parser.add_argument("-algorithm", default='FedPost', type=str)

    parser.add_argument("-learning_rate", default=3e-4, type=float)  # 5e-5 follow 邱锡鹏, 2e-5 follow MTC, 3e-4 for IMG_CLF follow https://arxiv.org/pdf/2402.15638, 3e-4 for Tabular_CLF
    parser.add_argument("-optimize_method", default='sgd', type=str)
    parser.add_argument("-model_type", default='ANN', type=str, choices=['ANN', 'LogisticRegression'])
    parser.add_argument("-dataset", default='ADULT', type=str, choices=['ADULT', 'COMPAS', 'DRUG', 'DUTCH'])
    parser.add_argument("-task", default='Tabular_CLF', type=str, choices=['SENT_CLF', 'IMG_CLF', "Tabular_CLF"])
    parser.add_argument("-batch_size", default=256, type=int, help="batch size")
    parser.add_argument("-test_batch_size", default=256, type=int, help="test batch size")
    parser.add_argument("-cuda", default="0,1,2,3", type=str, help="cuda")
    parser.add_argument("-max_len", default=128, type=int, help="text length to chunk")
    parser.add_argument("-system_data_count", default=None, type=int,
                        help="Limit the total number of training samples used in the experiment. "
                             "When set to a positive integer N, only the first N samples are used. "
                             "Used for quick testing/smoke tests. Default: None (use all data). "
                             "限制实验使用的训练样本总数。设为正整数N时，仅使用前N条样本，用于快速测试/冒烟测试。默认None（使用全部数据）")
    parser.add_argument("-tb_monitor", default=None, type=str,
                        help="TensorBoard monitoring configuration in JSON format. "
                             "Available options: test(bool), gradient(bool), embedding(bool), "
                             "fisher(bool), sharpness(bool), activation(bool), update_stats(bool), "
                             "client_divergence(bool), and their *_freq(int) counterparts. "
                             "Example: '{\"gradient\":false,\"gradient_freq\":10}' to disable gradient monitoring. "
                             "TensorBoard监控配置（JSON格式）。可用选项：test、gradient、embedding、fisher、sharpness、activation、update_stats、client_divergence（布尔值），"
                             "以及对应的 *_freq（频率，整数）。示例：'{\"gradient\":false,\"gradient_freq\":10}' 禁用梯度监控")
    parser.add_argument("-tb_log_dir", default=None, type=str,
                        help="Base directory for TensorBoard logs. "
                             "If specified, logs will be saved under this directory with structure: <tb_log_dir>/<dataset>/<algorithm>/<experiment_name>_<timestamp>. "
                             "Default: None (uses ./tensorboard_log). "
                             "TensorBoard日志的基础目录。如果指定，日志将按以下结构保存：<tb_log_dir>/<dataset>/<algorithm>/<experiment_name>_<timestamp>。默认None(使用./tensorboard_log)")
    parser.add_argument("-model_heter_frac", default=0.5, type=float,
                        help="Fraction of clients with heterogeneous models (0-1). "
                             "Only applies to Progressive algorithms. Default: 0.5. "
                             "模型异构客户端的比例(0-1)。仅对Progressive系列算法生效。默认0.5")
    parser.add_argument("-split_strategy", default=None, type=str,
                        help="Data splitting strategy: Dirichlet01, Dirichlet05, Dirichlet1, or Uniform. "
                             "Dirichlet01=high heterogeneity, Uniform=balanced. Default: None (use all strategies). "
                             "数据划分策略：Dirichlet01(高异构), Dirichlet05(中异构), Dirichlet1(低异构), Uniform(均匀)。默认None(使用全部策略)")
    parser.add_argument("-communication_round_I", default=None, type=int,
                        help="Number of communication rounds. If specified, overrides the default value. "
                             "Default: None (use value from epoch_T_communication_I_list). "
                             "通信轮次数。如果指定，覆盖默认值。默认None(使用epoch_T_communication_I_list中的值)")
    parser.add_argument("-algorithm_epoch_T", default=None, type=int,
                        help="Number of local training epochs. If specified, overrides the default value. "
                             "Default: None (use value from epoch_T_communication_I_list).")
    parser.add_argument("-num_clients_K", default=None, type=int,
                        help="Number of clients. If specified, overrides the default value. "
                             "Default: None (use values from num_clients_K_list). "
                             "客户端数量。如果指定，覆盖默认值。默认None(使用num_clients_K_list中的值)")
    parser.add_argument("-start_exp", default=1, type=int, help="Start from experiment number (1-12)")
    add_experiment_state_arguments(parser)

    args = parser.parse_args()
    param_dict = vars(args)
    param_dict["CUDA_VISIBLE_DEVICES"] = param_dict["cuda"]
    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['CUDA_VISIBLE_DEVICES']
    
    if param_dict.get('tb_monitor') is not None:
        try:
            param_dict['tb_monitor'] = json.loads(param_dict['tb_monitor'])
        except json.JSONDecodeError:
            print(f"[WARNING] Invalid tb_monitor JSON format: {param_dict['tb_monitor']}")
            param_dict['tb_monitor'] = {}
    
    return param_dict


def main(dataset_name, algorithm, hypothesis, classifier_type, device, param_dict):
    from algorithm.fedfact_core import validate_fedfact_entrypoint
    validate_fedfact_entrypoint(algorithm, "Tabular_CLF")
    import time
    
    dataset_name_list = dataset_name.split(",")
    for dataset_name in dataset_name_list:
        dataset_name = dataset_name.strip()
        if os.path.exists(os.path.join("./json/dataset/", dataset_name + ".json")):
            with open(os.path.join("./json/dataset/", dataset_name + ".json"), "r") as f:
                temp_dict = json.load(f)
            param_dict.update(**temp_dict)

    if os.path.exists(os.path.join("./json/algorithm/", algorithm + ".json")):
        with open(os.path.join("./json/algorithm/", algorithm + ".json"), "r") as f:
            temp_dict = json.load(f)
        param_dict.update(**temp_dict)

    import torch
    if "gpu" in device.lower():
        param_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        param_dict['device'] = "cpu"

    param_dict['device'] = device

    if param_dict.get('split_strategy') is not None:
        split_strategy_list = [param_dict['split_strategy']]
    else:
        split_strategy_list = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
    FL_drop_rate_list = [0]
    param_dict['dataset_name'] = dataset_name

    if param_dict.get('communication_round_I') is not None:
        epoch_T = param_dict['algorithm_epoch_T'] if param_dict['algorithm_epoch_T'] is not None else 2
        epoch_T_communication_I_list = [(epoch_T, param_dict['communication_round_I'])]
    else:
        epoch_T_communication_I_list = [(2, 500)]
    fraction_list = [0.1]
    
    if param_dict.get('num_clients_K') is not None:
        num_clients_K_list = [param_dict['num_clients_K']]
    else:
        num_clients_K_list = [20, 30, 40]

    model_heter_frac_list = [0]
    if "Progressive".lower() in algorithm.lower():
        model_heter_frac_list = [0, 0.5, 1]

    param_dict['dataset_name'] = dataset_name
    if "Centralized".lower() in algorithm.lower():
        split_strategy_list = ["Uniform"]
        num_clients_K_list = [1]
        epoch_T_communication_I_list = [(1, 1)]

    param_dict['model_type'] = 'ANN'
    param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis
    param_dict['classifier_type'] = classifier_type
    param_dict['miu'] = 1
    param_dict['γ_k_style'] = "uniform_client"
    tolerance_rate = 1
    param_dict['FedRenyi_λ'] = 1
    param_dict['global_group_loss_gap'] = 0.1

    total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list) * len(
        fraction_list) * len(num_clients_K_list) * len(model_heter_frac_list)

    param_dict['one_batch_per_Epoch'] = False

    start_exp = param_dict.get('start_exp', 1)

    current_exp = 1
    for split_strategy in split_strategy_list:
        for model_heter_frac in model_heter_frac_list:
            param_dict['model_heter_frac'] = model_heter_frac
            for FL_drop_rate in FL_drop_rate_list:
                param_dict['FL_drop_rate'] = FL_drop_rate
                for algorithm_epoch_T, communication_round_I in epoch_T_communication_I_list:
                    for fraction in fraction_list:
                        for num_clients_K in num_clients_K_list:
                            param_dict['split_strategy'] = split_strategy
                            param_dict['num_clients_K'] = num_clients_K
                            param_dict['algorithm_epoch_T'] = algorithm_epoch_T
                            param_dict['communication_round_I'] = communication_round_I
                            param_dict['FL_fraction'] = fraction
                            param_dict['tolerance_τ'] = int(tolerance_rate * algorithm_epoch_T * communication_round_I)

                            log_path = os.path.join("./log_path", param_dict['dataset_name'],
                                                    param_dict['split_strategy'],
                                                    param_dict['algorithm'],
                                                    param_dict['hypothesis'],
                                                    str(num_clients_K) + "Clients")
                            check_and_make_the_path(log_path)
                            log_file = os.path.join(log_path, str(current_exp) + ".txt")

                            if current_exp < start_exp:
                                print(f"  [SKIP] Experiment {current_exp}/{total_Experiment_NO} - before start_exp")
                                current_exp += 1
                                continue

                            param_dict['log_path'] = log_file
                            file_handler = logging.FileHandler(log_file, encoding='utf-8')
                            file_handler.setFormatter(formatter)
                            logger.addHandler(file_handler)

                            result_path = os.path.join("./result_path", param_dict['dataset_name'],
                                                    param_dict['split_strategy'],
                                                    param_dict['algorithm'],
                                                    param_dict['hypothesis'],
                                                    str(num_clients_K) + "Clients")
                            param_dict['basic_path'] = result_path

                            check_and_make_the_path(result_path)
                            result_path = os.path.join(result_path, str(current_exp) + ".txt")
                            param_dict['result_path'] = result_path

                            model_path = os.path.join("./save_path", param_dict['dataset_name'],
                                                      param_dict['split_strategy'],
                                                      param_dict['algorithm'],
                                                      param_dict['hypothesis'],
                                                      str(num_clients_K) + "Clients")
                            check_and_make_the_path(model_path)
                            param_dict['model_path'] = model_path
                            for k in range(param_dict["num_clients_K"]):
                                _ = os.path.join(model_path, "client_" + str(k + 1))
                                check_and_make_the_path(_)
                            logger.info(f"Experiment {current_exp}/{total_Experiment_NO} setup finish")
                            param_dict['Experiment_NO'] = str(current_exp)

                            logger.info("Parameter announcement")
                            for para_key in list(param_dict.keys()):
                                if "_common" in para_key:
                                    continue
                                logger.info(f"****** {para_key} : {param_dict[para_key]} ******")
                            logger.info("-----------------------------------------------------------------------------")

                            torch.cuda.empty_cache()
                            Experiment(param_dict)
                            torch.cuda.empty_cache()

                            current_exp += 1
                            logger.removeHandler(file_handler)
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")


if __name__ == '__main__':
    param_dict = Argparse()

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    main(dataset_name=param_dict['dataset'],
         algorithm=param_dict['algorithm'],
         hypothesis=param_dict['model_type'],
         classifier_type="linear",
         device=_device,
         param_dict=param_dict)
