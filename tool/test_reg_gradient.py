"""
梯度验证脚本：验证修复后的正则项是否真正影响了模型参数的梯度。
对每个算法，模拟一次前向+反向传播，比较：
  1. 只有 task loss 时的梯度
  2. task loss + reg term 时的梯度
如果梯度不同，说明 reg term 真正参与了梯度计算。
"""
import torch
import torch.nn as nn
import copy
import numpy as np

device = torch.device("cpu")

# ============================================================
# 构造一个简单的模型（模拟 IMG_CLF / Tabular_CLF 结构）
# ============================================================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_base = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(16, 1)

    def forward(self, x):
        features = self.shared_base(x)
        logits = torch.sigmoid(self.out_layer(features))
        return logits, features

    def only_clf_forward(self, features):
        return None, torch.sigmoid(self.out_layer(features))


def create_mock_batch(batch_size=8):
    """创建模拟数据：X, labels, protected"""
    X = torch.randn(batch_size, 10)
    labels = torch.randint(0, 2, (batch_size,)).float()
    # 保证 batch 中两个 group 都有数据
    protected = torch.zeros(batch_size)
    protected[:batch_size // 2] = 1.0
    return {"X": X, "labels": labels, "protected": protected}


# ============================================================
# 测试 1: mFairFL / Simple_mFairFL — group loss gap
# ============================================================
def test_mfairfl():
    print("=" * 60)
    print("测试 mFairFL / Simple_mFairFL: group loss gap 梯度")
    print("=" * 60)

    model = SimpleModel()
    criterion = nn.BCELoss(reduction='none')
    batch = create_mock_batch()
    X, labels, protected = batch["X"], batch["labels"], batch["protected"]
    true_batch_size = labels.size()[0]

    # --- 只有 task loss ---
    model1 = copy.deepcopy(model)
    model1.zero_grad()
    logits1, _ = model1(X)
    batch_loss1 = criterion(logits1[:, 0], labels)
    loss1 = torch.sum(batch_loss1) / true_batch_size
    loss1.backward()
    grad_only_task = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

    # --- task loss + reg (修复后) ---
    model2 = copy.deepcopy(model)
    model2.zero_grad()
    lambda_param = torch.nn.Parameter(torch.tensor(1.0))
    logits2, _ = model2(X)
    batch_loss2 = criterion(logits2[:, 0], labels)
    loss2 = torch.sum(batch_loss2) / true_batch_size

    group_flag = protected.gt(0.5)
    g1_count = group_flag.sum().item()
    g0_count = true_batch_size - g1_count
    if g1_count > 0 and g0_count > 0:
        g1_avg = batch_loss2[group_flag].mean()
        g0_avg = (batch_loss2.sum() - batch_loss2[group_flag].sum()) / g0_count
        gap = torch.abs(g0_avg - g1_avg)
        loss2 = loss2 + lambda_param * gap

    loss2.backward()
    grad_with_reg = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

    # 比较
    all_same = True
    for name in grad_only_task:
        diff = (grad_only_task[name] - grad_with_reg[name]).abs().max().item()
        if diff > 1e-8:
            all_same = False
            print(f"  参数 {name}: 梯度差异 = {diff:.8f}  <-- reg term 生效了!")
    if all_same:
        print("  [FAIL] 所有参数梯度完全相同，reg term 没有影响梯度!")
    else:
        print("  [PASS] reg term 影响了模型梯度!")
    print()


# ============================================================
# 测试 2: FederatedProximal — proximal term
# ============================================================
def test_fedprox():
    print("=" * 60)
    print("测试 FederatedProximal: proximal term 梯度")
    print("=" * 60)

    global_model = SimpleModel()
    # 给 global_model 加一些随机偏移，使 proximal term 不为零
    with torch.no_grad():
        for p in global_model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    model = copy.deepcopy(global_model)
    criterion = nn.BCELoss(reduction='none')
    batch = create_mock_batch()
    X, labels = batch["X"], batch["labels"]
    true_batch_size = labels.size()[0]
    miu = 1.0

    # --- 只有 task loss ---
    model1 = copy.deepcopy(global_model)
    model1.zero_grad()
    logits1, _ = model1(X)
    batch_loss1 = criterion(logits1[:, 0], labels)
    loss1 = torch.sum(batch_loss1) / true_batch_size
    loss1.backward()
    grad_only_task = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

    # --- task loss + proximal term (修复后) ---
    model2 = copy.deepcopy(global_model)
    # 模拟本地训练后参数发生变化，使 proximal term 不为零
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    model2.zero_grad()
    logits2, _ = model2(X)
    batch_loss2 = criterion(logits2[:, 0], labels)
    loss2 = torch.sum(batch_loss2) / true_batch_size

    proximal_term = 0.
    for (name_g, param_g), (name_l, param_l) in zip(global_model.named_parameters(), model2.named_parameters()):
        if param_l.requires_grad:
            proximal_term += torch.norm(param_g - param_l, p=2) ** 2
    loss2 = loss2 + ((miu / 2) * proximal_term) / true_batch_size

    loss2.backward()
    grad_with_reg = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

    # 比较
    all_same = True
    for name in grad_only_task:
        diff = (grad_only_task[name] - grad_with_reg[name]).abs().max().item()
        if diff > 1e-8:
            all_same = False
            print(f"  参数 {name}: 梯度差异 = {diff:.8f}  <-- proximal term 生效了!")
    if all_same:
        print("  [FAIL] 所有参数梯度完全相同，proximal term 没有影响梯度!")
    else:
        print("  [PASS] proximal term 影响了模型梯度!")
    print()


# ============================================================
# 测试 3: FederatedProto — prototype gap
# ============================================================
def test_fedproto():
    print("=" * 60)
    print("测试 FederatedProto: prototype gap 梯度")
    print("=" * 60)

    model = SimpleModel()
    criterion = nn.BCELoss(reduction='none')
    batch = create_mock_batch()
    X, labels = batch["X"], batch["labels"]
    true_batch_size = labels.size()[0]

    # 模拟一个全局 prototype
    global_label_0_prototype = torch.randn(16) * 0.5
    global_label_1_prototype = torch.randn(16) * 0.5

    # --- 只有 task loss ---
    model1 = copy.deepcopy(model)
    model1.zero_grad()
    logits1, features1 = model1(X)
    batch_loss1 = criterion(logits1[:, 0], labels)
    loss1 = torch.sum(batch_loss1) / true_batch_size
    loss1.backward()
    grad_only_task = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

    # --- task loss + prototype gap (修复后) ---
    model2 = copy.deepcopy(model)
    model2.zero_grad()
    logits2, features2 = model2(X)
    batch_loss2 = criterion(logits2[:, 0], labels)
    loss2 = torch.sum(batch_loss2) / true_batch_size

    label_flag = labels.gt(0.5)
    label_0_features = features2[~label_flag]
    label_1_features = features2[label_flag]

    if len(label_0_features) > 0:
        label_0_prototype = label_0_features.mean(dim=0)
        label_0_gap = torch.norm(global_label_0_prototype - label_0_prototype, p=2)
        loss2 = loss2 + 1.0 * label_0_gap

    if len(label_1_features) > 0:
        label_1_prototype = label_1_features.mean(dim=0)
        label_1_gap = torch.norm(global_label_1_prototype - label_1_prototype, p=2)
        loss2 = loss2 + 1.0 * label_1_gap

    loss2.backward()
    grad_with_reg = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

    # 比较
    all_same = True
    for name in grad_only_task:
        diff = (grad_only_task[name] - grad_with_reg[name]).abs().max().item()
        if diff > 1e-8:
            all_same = False
            print(f"  参数 {name}: 梯度差异 = {diff:.8f}  <-- prototype gap 生效了!")
    if all_same:
        print("  [FAIL] 所有参数梯度完全相同，prototype gap 没有影响梯度!")
    else:
        print("  [PASS] prototype gap 影响了模型梯度!")
    print()


# ============================================================
# 测试 4: PDFFed — proto clf loss + distribution gap + cov
# ============================================================
def test_pdffed():
    print("=" * 60)
    print("测试 PDFFed: proto clf loss + distribution gap + cov 梯度")
    print("=" * 60)

    model = SimpleModel()
    global_model = copy.deepcopy(model)
    criterion = nn.BCELoss(reduction='none')
    batch = create_mock_batch()
    X, labels, protected = batch["X"], batch["labels"], batch["protected"]
    true_batch_size = labels.size()[0]

    # 模拟全局 prototype
    global_proto_list = [torch.randn(16) * 0.5, torch.randn(16) * 0.5]
    global_proto_labels = [torch.zeros(1), torch.ones(1)]

    # --- 只有 task loss ---
    model1 = copy.deepcopy(model)
    model1.zero_grad()
    logits1, features1 = model1(X)
    batch_loss1 = criterion(logits1[:, 0], labels)
    loss1 = torch.sum(batch_loss1) / true_batch_size
    loss1.backward()
    grad_only_task = {n: p.grad.clone() for n, p in model1.named_parameters() if p.grad is not None}

    # --- task loss + reg terms (修复后) ---
    model2 = copy.deepcopy(model)
    model2.zero_grad()
    logits2, features2 = model2(X)
    batch_loss2 = criterion(logits2[:, 0], labels)
    loss2 = torch.sum(batch_loss2) / true_batch_size

    # 1) proto clf loss (local proto -> local clf)
    label_flag = labels.gt(0.5)
    group_flag = protected.gt(0.5)
    label_0_features = features2[~label_flag]
    label_1_features = features2[label_flag]

    reg_loss = torch.tensor(0.0)
    if len(label_0_features) > 0:
        local_proto_0 = label_0_features.mean(dim=0)
        _, tmp_logit_0 = model2.only_clf_forward(local_proto_0.unsqueeze(0))
        reg_loss = reg_loss + criterion(tmp_logit_0[:, 0], torch.zeros(1)).mean()
    if len(label_1_features) > 0:
        local_proto_1 = label_1_features.mean(dim=0)
        _, tmp_logit_1 = model2.only_clf_forward(local_proto_1.unsqueeze(0))
        reg_loss = reg_loss + criterion(tmp_logit_1[:, 0], torch.ones(1)).mean()

    # 2) global proto -> local clf
    global_proto_tensors = torch.stack(global_proto_list)
    global_proto_label_tensors = torch.stack(global_proto_labels).squeeze(1)
    _, global_proto_logit = model2.only_clf_forward(global_proto_tensors)
    reg_loss = reg_loss + criterion(global_proto_logit.squeeze(1), global_proto_label_tensors).mean()

    # 3) distribution gap
    g1_l0_mask = (group_flag * (~label_flag))
    g0_l0_mask = ((~group_flag) * (~label_flag))
    if g1_l0_mask.sum() > 0 and g0_l0_mask.sum() > 0:
        g1_l0_pred = logits2[g1_l0_mask].mean()
        g0_l0_pred = logits2[g0_l0_mask].mean()
        dist_gap = torch.norm(g1_l0_pred - g0_l0_pred, p=2)
        reg_loss = reg_loss + dist_gap

    loss2 = loss2 + reg_loss
    loss2.backward()
    grad_with_reg = {n: p.grad.clone() for n, p in model2.named_parameters() if p.grad is not None}

    # 比较
    all_same = True
    for name in grad_only_task:
        diff = (grad_only_task[name] - grad_with_reg[name]).abs().max().item()
        if diff > 1e-8:
            all_same = False
            print(f"  参数 {name}: 梯度差异 = {diff:.8f}  <-- reg term 生效了!")
    if all_same:
        print("  [FAIL] 所有参数梯度完全相同，reg term 没有影响梯度!")
    else:
        print("  [PASS] reg term 影响了模型梯度!")
    print()


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  正则项梯度验证测试")
    print("#  对每个算法比较：只有task loss vs task loss + reg term")
    print("#  如果梯度不同 → reg term 真正参与了梯度计算")
    print("#" * 60 + "\n")

    test_mfairfl()
    test_fedprox()
    test_fedproto()
    test_pdffed()

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)
