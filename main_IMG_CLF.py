import os
import json
import argparse
import re
import numpy as np

from tool.logger import *
from tool.utils import check_and_make_the_path
from experiment import Experiment


def analyze_experiment_log(log_file):
    """分析实验日志，返回已完成的测试次数和是否有最终汇总"""
    if not os.path.exists(log_file):
        return 0, False
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        test_count = len(re.findall(r'Trained Global Model Testing', content))
        has_summary = 'Mean' in content and 'STD' in content
        
        return test_count, has_summary
    except:
        return 0, False


def calculate_and_append_summary(log_file, algorithm):
    """从日志中提取3次测试结果，计算均值和标准差并追加到日志"""
    import time
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
    parser.add_argument("-algorithm", default='PDFFed', type=str)

    parser.add_argument("-learning_rate", default=3e-4, type=float)
    parser.add_argument("-optimize_method", default='sgd', type=str)
    parser.add_argument("-dataset", default='LFWA+', type=str, choices=['CelebA','UTKFace', 'FairFace', 'LFWA+'])
    parser.add_argument("-task", default='IMG_CLF', type=str, choices=['SENT_CLF', 'IMG_CLF', "Tabular_CLF"])

    parser.add_argument("-batch_size", default=256, type=int, help="batch size")
    parser.add_argument("-test_batch_size", default=256, type=int, help="test batch size")
    parser.add_argument("-cuda", default="0,1,2,3", type=str, help="cuda")
    parser.add_argument("-max_len", default=128, type=int, help="text length to chunk")
    parser.add_argument("-system_data_count", default=None)
    parser.add_argument("-model_heter_frac", default=0.5)
    parser.add_argument("-split_strategy", default=None, type=str)
    parser.add_argument("-communication_round_I", default=None, type=int)
    parser.add_argument("-start_exp", default=1, type=int, help="Start from experiment number (1-12)")
    parser.add_argument("-resume", action='store_true', help="Auto-resume from the first incomplete experiment")
    parser.add_argument("-exp_repeat_times", type=int, default=3,
                        help="Number of times to repeat each experiment with different seeds for statistical significance. "
                             "Default: 3. Results are reported as Mean +/- STD across repeats. "
                             "每个实验用不同随机种子重复运行的次数，用于统计显著性。默认3次，结果报告为 Mean +/- STD")
    parser.add_argument("-parallel_repeats", type=int, default=1,
                        help="Each experiment repeats 3 times with different seeds. "
                             "This param controls how many repeat runs execute in parallel via multiprocessing. "
                             "1=serial (default), 2=two repeats in parallel, 3=all three repeats in parallel. "
                             "每个实验会重复3次（不同随机种子），此参数控制几次重复同时并行执行："
                             "1=串行（默认），2=两次并行，3=三次全部并行")

    args = parser.parse_args()
    param_dict = vars(args)
    param_dict["CUDA_VISIBLE_DEVICES"] = param_dict["cuda"]
    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['CUDA_VISIBLE_DEVICES']
    return param_dict


def main(dataset_name, algorithm, hypothesis, classifier_type, device, param_dict):
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


    split_strategy_list = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]

    FL_drop_rate_list = [0]
    param_dict['dataset_name'] = dataset_name

    epoch_T_communication_I_list = [(2, 5)]
    fraction_list = [0.1]
    num_clients_K_list = [20, 30, 40]

    model_heter_frac_list = [0]
    if "Progressive".lower() in algorithm.lower():
        model_heter_frac_list = [0, 0.5, 1]

    param_dict['dataset_name'] = dataset_name
    if "Centralized".lower() in algorithm.lower():
        split_strategy_list = ["Uniform"]
        num_clients_K_list = [1]
        epoch_T_communication_I_list = [(1, 1)]

    param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis
    param_dict['classifier_type'] = classifier_type
    param_dict['miu'] = 1
    param_dict['γ_k_style'] = "uniform_client"
    tolerance_rate = 1
    param_dict['FedRenyi_λ'] = 1

    param_dict['global_group_loss_gap'] = 0.1

    Experiment_NO = 1
    total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list) * len(
        fraction_list) * len(num_clients_K_list) * len(model_heter_frac_list)

    param_dict['one_batch_per_Epoch'] = False

    start_exp = param_dict.get('start_exp', 1)
    resume_mode = param_dict.get('resume', False)

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
                            log_file = os.path.join(log_path, str(Experiment_NO) + ".txt")

                            if resume_mode:
                                test_count, has_summary = analyze_experiment_log(log_file)
                                
                                if test_count >= 3 and has_summary:
                                    print(f"  [SKIP] Experiment {Experiment_NO}/{total_Experiment_NO} - already complete")
                                    Experiment_NO += 1
                                    continue
                                elif test_count >= 3 and not has_summary:
                                    print(f"  [SUMMARY] Experiment {Experiment_NO}/{total_Experiment_NO} - calculating summary...")
                                    calculate_and_append_summary(log_file, algorithm)
                                    Experiment_NO += 1
                                    continue
                                elif 0 < test_count < 3:
                                    print(f"  [RESUME] Experiment {Experiment_NO}/{total_Experiment_NO} - has {test_count}/3 tests")
                                else:
                                    print(f"  [START] Experiment {Experiment_NO}/{total_Experiment_NO} - starting fresh")
                                resume_mode = False
                            
                            if Experiment_NO < start_exp:
                                print(f"  [SKIP] Experiment {Experiment_NO}/{total_Experiment_NO} - before start_exp")
                                Experiment_NO += 1
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
                            result_path = os.path.join(result_path, str(Experiment_NO) + ".txt")
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
                            logger.info(f"Experiment {Experiment_NO}/{total_Experiment_NO} setup finish")
                            param_dict['Experiment_NO'] = str(Experiment_NO)

                            logger.info("Parameter announcement")
                            for para_key in list(param_dict.keys()):
                                if "_common" in para_key:
                                    continue
                                logger.info(f"****** {para_key} : {param_dict[para_key]} ******")
                            logger.info("-----------------------------------------------------------------------------")

                            torch.cuda.empty_cache()
                            Experiment(param_dict)
                            torch.cuda.empty_cache()

                            Experiment_NO += 1
                            logger.removeHandler(file_handler)
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")


if __name__ == '__main__':
    param_dict = Argparse()

    _device = "cpu" if not param_dict["cuda"] else "cuda"
    for algorithm in [param_dict['algorithm']]:
        for dataset in [param_dict['dataset']]:
            main(dataset_name=dataset,
                 algorithm=algorithm,
                 hypothesis="BERTCLASSIFIER",
                 classifier_type="linear",
                 device=_device,
                 param_dict=param_dict)