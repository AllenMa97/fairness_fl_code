import os
import json
import argparse

from tool.logger import *
from tool.utils import check_and_make_the_path
from experiment import Experiment


def Argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'test'])
    # parser.add_argument("-algorithm", default='Centralized', type=str)
    # parser.add_argument("-algorithm", default='Separate', type=str)
    # parser.add_argument("-algorithm", default='FedAvg', type=str)
    # parser.add_argument("-algorithm", default='FedProx', type=str)
    # parser.add_argument("-algorithm", default='Scaffold', type=str)
    # parser.add_argument("-algorithm", default='FedRep', type=str)
    # parser.add_argument("-algorithm", default='FedNova', type=str)
    # parser.add_argument("-algorithm", default='FedProto', type=str)
    # parser.add_argument("-algorithm", default='OSFL', type=str)
    # parser.add_argument("-algorithm", default='CO_BOOSTING', type=str)
    # parser.add_argument("-algorithm", default='FairFed', type=str)
    # parser.add_argument("-algorithm", default='FedFair', type=str)
    # parser.add_argument("-algorithm", default='FL_FairBatch', type=str)
    # parser.add_argument("-algorithm", default='FedFB', type=str)
    # parser.add_argument("-algorithm", default='FedRenyi', type=str)
    # parser.add_argument("-algorithm", default='FedPost', type=str)
    # parser.add_argument("-algorithm", default='FedMZY', type=str)
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


    parser.add_argument("-learning_rate", default=5e-5, type=float)  # 5e-5 follow 邱锡鹏, 2e-5 follow MTC
    parser.add_argument("-optimize_method", default='sgd', type=str)
    # parser.add_argument("-dataset", default='bios', type=str, choices=['moji', 'bios', 'CelebA'])
    parser.add_argument("-dataset", default='moji', type=str, choices=['moji', 'bios', 'CelebA'])
    # parser.add_argument("-task", default='SENT_CLF', type=str, choices=['SENT_CLF', 'IMG_CLF', "Tabular_CLF"]) # 句子分类 or 图像分类
    parser.add_argument("-task", default='SENT_CLF', type=str, choices=['SENT_CLF', 'IMG_CLF', "Tabular_CLF"]) # 句子分类 or 图像分类 or 表格数据分类

    parser.add_argument("-batch_size", default=64, type=int, help="batch size") # 图像分类3060显卡极限1024 or 句子分类256
    parser.add_argument("-test_batch_size", default=256, type=int, help="test batch size")
    parser.add_argument("-cuda", default="0,1,2,3", type=str, help="cuda")
    parser.add_argument("-max_len", default=128, type=int, help="text length to chunk")
    parser.add_argument("-system_data_count", default=4000)
    parser.add_argument("-model_heter_frac", default=0.5)

    args = parser.parse_args()
    param_dict = vars(args)
    param_dict["CUDA_VISIBLE_DEVICES"] = param_dict["cuda"]
    os.environ["CUDA_VISIBLE_DEVICES"] = param_dict['CUDA_VISIBLE_DEVICES']
    return param_dict


def main(dataset_name, algorithm, hypothesis, classifier_type, device, param_dict):
    # Dataset Hyper-params
    dataset_name_list = dataset_name.split(",")
    for dataset_name in dataset_name_list:
        dataset_name = dataset_name.strip()
        if os.path.exists(os.path.join("./json/dataset/", dataset_name + ".json")):
            with open(os.path.join("./json/dataset/", dataset_name + ".json"), "r") as f:
                temp_dict = json.load(f)
            param_dict.update(**temp_dict)
    # Algorithm Hyper-params
    if os.path.exists(os.path.join("./json/algorithm/", algorithm + ".json")):
        with open(os.path.join("./json/algorithm/", algorithm + ".json"), "r") as f:
            temp_dict = json.load(f)
        param_dict.update(**temp_dict)

    import torch
    if "gpu" in device.lower():
        param_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment
    else:
        param_dict['device'] = "cpu"

    param_dict['device'] = device

    split_strategy_list = ["Dirichlet01", "Dirichlet05", "Dirichlet1", "Uniform"]
    # split_strategy_list = ["Uniform","Dirichlet1"]
    # split_strategy_list = ["Dirichlet01"]
    # split_strategy_list = ["Uniform"]

    FL_drop_rate_list = [0]  # 设置掉线率
    param_dict['dataset_name'] = dataset_name

    epoch_T_communication_I_list = [(2, 5)]  # 本地走T个epoch后进行一次通信，共走T*I个epoch，每次聚合都做性能测试
    fraction_list = [0.1]
    num_clients_K_list = [20, 30, 40]  # 设置客户端数目
    # num_clients_K_list = [3, 30, 40]  # 设置客户端数目
    # num_clients_K_list = [20]  # 设置客户端数目

    model_heter_frac_list = [0]  # 设置模型异构客户的比例
    if "Progressive".lower() in algorithm.lower():
        model_heter_frac_list = [0, 0.5, 1]  # 设置模型异构客户的比例

    param_dict['dataset_name'] = dataset_name
    if "Centralized".lower() in algorithm.lower():
        split_strategy_list = ["Uniform"]
        num_clients_K_list = [1]  # 设置客户端数目
        epoch_T_communication_I_list = [(1, 1)]  # 本地走T个epoch后进行一次通信，共走T*I个epoch，每次聚合都做性能测试

    param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis
    param_dict['classifier_type'] = classifier_type
    param_dict['miu'] = 1 # FedProx的超参数
    param_dict['γ_k_style'] = "uniform_distribution" # FedRenyi的两个模式, uniform_distribution 或者 unifor_client
    tolerance_rate = 1  # FedRenyi的超参数
    param_dict['FedRenyi_λ'] = 1  # FedRenyi的超参数



    param_dict['global_group_loss_gap'] = 0.1  # Progressive的超参数
    # Serial number of experiment
    Experiment_NO = 1
    total_Experiment_NO = len(FL_drop_rate_list) * len(epoch_T_communication_I_list) * len(split_strategy_list) * len(
        fraction_list) * len(num_clients_K_list) * len(model_heter_frac_list)

    param_dict['one_batch_per_Epoch'] = False

    # Main Loop
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

                            ################################################################################################
                            # Create the log
                            log_path = os.path.join("./log_path", param_dict['dataset_name'],
                                                    param_dict['split_strategy'],
                                                    param_dict['algorithm'],
                                                    param_dict['hypothesis'],
                                                    str(num_clients_K) + "Clients")
                            check_and_make_the_path(log_path)
                            log_path = os.path.join(log_path, str(Experiment_NO) + ".txt")
                            param_dict['log_path'] = log_path
                            file_handler = logging.FileHandler(log_path, encoding='utf-8')
                            file_handler.setFormatter(formatter)
                            logger.addHandler(file_handler)
                            ################################################################################################
                            # Create the result path
                            result_path = os.path.join("./result_path", param_dict['dataset_name'],
                                                    param_dict['split_strategy'],
                                                    param_dict['algorithm'],
                                                    param_dict['hypothesis'],
                                                    str(num_clients_K) + "Clients")
                            param_dict['basic_path'] = result_path

                            check_and_make_the_path(result_path)
                            result_path = os.path.join(result_path, str(Experiment_NO) + ".txt")
                            param_dict['result_path'] = result_path
                            ################################################################################################
                            # Create the model path
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
                            ################################################################################################
                            # Parameter announcement
                            logger.info("Parameter announcement")
                            for para_key in list(param_dict.keys()):
                                if "_common" in para_key:
                                    continue
                                logger.info(f"****** {para_key} : {param_dict[para_key]} ******")
                            logger.info("-----------------------------------------------------------------------------")
                            ################################################################################################
                            # Experiment
                            torch.cuda.empty_cache()
                            Experiment(param_dict)
                            torch.cuda.empty_cache()
                            # Clear the saved path's pt files
                            # shutil.rmtree(model_path)
                            Experiment_NO += 1
                            logger.removeHandler(file_handler)
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                            logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")


if __name__ == '__main__':
    param_dict = Argparse()
    main(dataset_name=param_dict['dataset'],
         algorithm=param_dict['algorithm'],
         hypothesis="BERTCLASSIFIER",
         classifier_type="linear",
         device="cuda",
         # device="cpu",
         param_dict=param_dict)
