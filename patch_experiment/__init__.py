"""Phase A 机制验证实验（埋点与队列生成）。

本目录只放"验证方法论"的代码，不修改算法本体：
- probe.py            机制变量采集（Δ_rep / 轨迹），由 PDFFED_PROBE=1 打开
- gen_phase_a.py      生成 COMPAS / DUTCH 的统一网格队列
- run_queue.sh        队列执行器（带埋点环境变量）

启用方式（不改训练逻辑）：
    PDFFED_PROBE=1 python main_Tabular_CLF.py ... 其余参数不变
"""
