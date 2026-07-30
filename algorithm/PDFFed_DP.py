# PDFFed_DP: PDFFed with Differential Privacy (对应挑战1 / 定理5)
#
# 本文件是 PDFFed 的差分隐私变体入口。主算法逻辑完全复用 PDFFed.PDF_Fed，
# 仅通过 param_dict 开启 LDP 噪声注入开关，在客户端上传原型前添加高斯噪声，
# 使聚合后的全局原型满足分布式差分隐私。
#
# 设计原则：
#   - 零代码重复：主流程复用 PDFFed.PDF_Fed，不重写
#   - 职责分离：本文件只负责 DP 参数配置和入口包装
#   - 可消融对比：PDFFed vs PDFFed_DP 可直接对比隐私-效用权衡
#
# 定理5 设定（proof_zh.md）：
#   客户端 k 上传原型前添加 ε-LDP 噪声 ξ_k ~ N(0, σ_noise²·I_d)
#   其中 σ_noise² ≥ 2d·ln(2/δ)/ε²
#   聚合后满足分布式差分隐私：Pr[A(μ_global)=t] ≤ e^ε·Pr[A(μ_global^{-i})=t] + δ
#
# 定理6（反面证明）：
#   传输敏感属性 → 成员推断攻击成功率 = 1 → 不满足 DP
#   PDFFed 选择传输原型而非敏感属性，加噪后满足 DP，从根本上避免此风险。

from algorithm.PDFFed import PDF_Fed


def PDF_Fed_DP(device,
               global_model,
               algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate,
               training_dataloaders,
               training_dataset,
               client_dataset_list,
               param_dict,
               testing_dataloader,
               testing_dataset_len,
               start_round=0):
    """
    PDFFed 的差分隐私变体。

    与 PDF_Fed 的唯一区别：在 param_dict 中开启 use_dp=True，
    并设置 DP 参数 epsilon 和 delta。主算法逻辑完全复用 PDF_Fed。

    参数说明（DP 专属，其余同 PDF_Fed）：
        param_dict['use_dp']: 是否启用 LDP 噪声（本函数强制为 True）
        param_dict['dp_epsilon']: 隐私预算 ε，越小隐私越强（默认 1.0）
        param_dict['dp_delta']: DP 失败概率 δ（默认 1e-5）
    """
    # 强制开启 DP 噪声注入
    param_dict['use_dp'] = True

    # DP 参数校验与默认值
    if 'dp_epsilon' not in param_dict:
        param_dict['dp_epsilon'] = 1.0
    if 'dp_delta' not in param_dict:
        param_dict['dp_delta'] = 1e-5

    # 调用主算法（DP 逻辑已在 _train_single_client_pdffed 的原型返回前注入）
    return PDF_Fed(device=device,
                   global_model=global_model,
                   algorithm_epoch_T=algorithm_epoch_T,
                   num_clients_K=num_clients_K,
                   communication_round_I=communication_round_I,
                   FL_fraction=FL_fraction,
                   FL_drop_rate=FL_drop_rate,
                   training_dataloaders=training_dataloaders,
                   training_dataset=training_dataset,
                   client_dataset_list=client_dataset_list,
                   param_dict=param_dict,
                   testing_dataloader=testing_dataloader,
                   testing_dataset_len=testing_dataset_len,
                   start_round=start_round)
