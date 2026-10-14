# PDFFed_DP: PDFFed with Differential Privacy (对应挑战1 / 定理10)
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
# 定理10 设定：
#   客户端 k 上传原型前添加 ε-LDP 噪声 ξ_k ~ N(0, σ_noise²·I_d)
#   聚合后满足分布式差分隐私：Pr[A(μ_global)=t] ≤ e^ε·Pr[A(μ_global^{-i})=t] + δ
#
#   噪声规模（本实现口径）：
#     (1) 样本级敏感度：原型是 m 个样本特征的均值，特征每维有界 B，
#         故 Δ₂ = B·√d / m；
#     (2) 组合（composition）：原型每轮通信上传一次，T 轮累积后按
#         DRV 强组合的完整形式记账
#           √(2T·ln(1/δ))·ε_round + T·ε_round·(e^{ε_round} − 1) = ε_total，
#         单轮预算 ε_round 取该等式在 [0, ε_total] 上的唯一正根；
#     (3) 高斯机制：σ = Δ₂·√(2·ln(2/δ)) / ε_round。
#   对外只暴露 dp_epsilon 作为**总预算**，单轮预算 ε_round 在内部折算。
#
# 定理11（反面证明）：
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
               start_round=0,
               resume_state=None):
    """
    PDFFed 的差分隐私变体。

    与 PDF_Fed 的唯一区别：在 param_dict 中开启 use_dp=True，
    并设置 DP 参数 epsilon 和 delta。主算法逻辑完全复用 PDF_Fed。

    参数说明（DP 专属，其余同 PDF_Fed）：
        param_dict['use_dp']: 是否启用 LDP 噪声（本函数强制为 True）
        param_dict['dp_epsilon']: 隐私预算 ε（**总预算**，整条训练过程的累积预算），
            越小隐私越强（默认 8.0）
        param_dict['dp_delta']: DP 失败概率 δ（默认 1e-5）
        param_dict['dp_feature_bound']: 特征每维取值上界 B（默认 1.0），
            地位等同于 DP-SGD 的裁剪界 C，是 DP 论证的前提假设。
            默认值依据实测锚点标定：BN+ReLU 后特征逐维幅值实测（B_obs）
            在 ADULT 上约 0.2，取 1.0 保留约 5 倍余量；B 必须是先验上界，
            若某数据集特征幅值更大，须上调本项否则敏感度被低估
        param_dict['dp_rounds']: 原型上传轮数 T（默认 communication_round_I）

    关于默认 ε 的取值 / On the default epsilon:
        噪声规模由三部分决定：
        1) 样本级敏感度：原型是 m 个样本特征的均值，特征每维有界 B，
           故 Δ₂ = B·√d / m，而非把每维敏感度都当作 1 得到的 √d；
        2) 组合（composition）：原型每轮通信上传一次，T 轮累积后按 DRV
           强组合的完整等式记账
             √(2T·ln(1/δ))·ε_round + T·ε_round·(e^{ε_round} − 1) = ε_total，
           单轮预算 ε_round 取该等式在 [0, ε_total] 上的唯一正根；
        3) 高斯机制：σ = Δ₂·√(2·ln(2/δ)) / ε_round。
        以 d=512、B=1、T=50、δ=1e-5、ε_total=8、m=2500 为例：
        组合因子 √(2T·ln(1/δ)) ≈ 33.93，解完整等式得 ε_round ≈ 0.1821；
        Δ₂ = 1·√512/2500 ≈ 0.00905 → σ ≈ 0.246，
        与原型坐标量级（BN+ReLU 后约 0.4）同阶，隐私-效用权衡可比。
        注意：若只反解第一项（ε_round = ε_total/33.93 ≈ 0.2358）会漏掉
        修正项 T·ε_round·(e^{ε_round}−1)，得到"比真实更私密"的乐观声称，
        故此处解完整等式；其代价是 σ 由 0.569 提高到 0.737（约 1.29 倍，
        B=3 口径下）。如需扫描隐私预算，直接覆盖 param_dict['dp_epsilon'] 即可。

    与旧版口径的差异 / Changes from the previous version:
        旧版 σ² ≥ 2d·ln(2/δ)/ε²，即把每维敏感度都当作 1（Δ=√d）且忽略组合，
        d=512、ε=8 时 σ≈13.97，比原型信号大一个数量级，DP 变体退化为随机原型。
        本版改用样本级敏感度 Δ₂=B·√d/m 并显式做组合折算，使对外暴露的
        ε_total 可与文献（DP-SGD 用 Δ=C，无 √d 因子）直接对齐。
    """
    # 强制开启 DP 噪声注入
    param_dict['use_dp'] = True

    # DP 参数校验与默认值
    if 'dp_epsilon' not in param_dict:
        param_dict['dp_epsilon'] = 8.0
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
                   start_round=start_round,
                   resume_state=resume_state)
