# -*- coding: utf-8 -*-
# FedSum 三任务冒烟测试：Tabular_CLF(ANN) / IMG_CLF(CNN) / SENT_CLF(桩BERT)
import sys, os, shutil, tempfile
sys.path.insert(0, '.')
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from hypothesis.ANNCLASSIFIER import RegularANN
from hypothesis.CNNCLASSIFIER import RegularCNN

tmp = tempfile.mkdtemp(prefix='fedsum_smoke_')


class StubBertClassifier(nn.Module):
    """桩 BERT 分类器：与 BertClassifier 接口一致（forward 返回 (feature, logits)）"""
    def __init__(self, n_classes=2, hidden=16, vocab=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.enc = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_classes)

    def only_PLM_forward(self, input_ids, attention_mask=None):
        feature = self.embed(input_ids).mean(dim=1)
        return self.enc(feature)

    def forward(self, input_ids, attention_mask=None):
        feature = self.only_PLM_forward(input_ids, attention_mask)
        return feature, self.out(feature)


def run_scenario(task, make_model, make_batch, collate, d_model=None):
    torch.manual_seed(0)
    np.random.seed(0)
    n, d = 32, 8
    if task == 'SENT_CLF':
        X = torch.randint(0, 1000, (n, 16))
    elif task == 'IMG_CLF':
        X = torch.randn(n, 3, 32, 32)
    else:
        X = torch.randn(n, d)
    y = torch.tensor(np.random.randint(0, 2, n), dtype=torch.long)

    class DS(Dataset):
        def __init__(self):
            self.X, self.y = X, y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            return make_batch(self.X[i], self.y[i])

    dataset = DS()
    train_set, test_set = random_split(dataset, [24, 8])
    client_dataset_list = random_split(train_set, [10, 8, 6])
    training_dataloaders = [DataLoader(c, batch_size=8, shuffle=True, collate_fn=collate) for c in client_dataset_list]
    testing_dataloader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collate)

    param_dict = {
        'task': task,
        'algorithm': 'FedSum',
        'batch_size': 8,
        'test_batch_size': 8,
        'optimize_method': 'adam',
        'learning_rate': 1e-3,
        'num_clients_K': 3,
        'communication_round_I': 1,
        'algorithm_epoch_T': 1,
        'FL_fraction': 1.0,
        'FL_drop_rate': 0.0,
        'emb_dim': d if d_model is None else d_model,
        'model_type': 'ANN',
        'use_amp': False,
        'device': 'cpu',
        'model_path': os.path.join(tmp, task, 'models'),
        'checkpoint_save_freq': 1,
        'checkpoint_keep_latest': 2,
        'tb_monitor': {k: False for k in ('test', 'system', 'gradient', 'embedding',
                                          'neural_collapse', 'fisher', 'sharpness',
                                          'activation', 'update_stats', 'client_divergence')},
    }
    os.makedirs(param_dict['model_path'], exist_ok=True)

    from algorithm.FederatedSum import Fed_Sum
    model = make_model()
    g, gpu_s, comm = Fed_Sum('cpu', model, 1, 3, 1, 1.0, 0.0,
                             training_dataloaders, train_set, client_dataset_list,
                             param_dict, testing_dataloader, len(test_set))
    assert g is not None and gpu_s > 0 and comm > 0, f"{task}: return values invalid"
    print(f"SMOKE_OK[{task}] gpu_seconds=%.4f comm=%.4f" % (gpu_s, comm))


# ---- Tabular_CLF（ANN）----
def tb_batch(X, y):
    return {'X': X, 'labels': y}

def tb_collate(batch):
    return {'X': torch.stack([b['X'] for b in batch]),
            'labels': torch.tensor([b['labels'] for b in batch], dtype=torch.long)}

run_scenario('Tabular_CLF', lambda: RegularANN(input_size=8), tb_batch, tb_collate)

# ---- IMG_CLF（RegularCNN）----
def im_batch(X, y):
    return {'img': X, 'labels': y}

def im_collate(batch):
    return {'img': torch.stack([b['img'] for b in batch]),
            'labels': torch.tensor([b['labels'] for b in batch], dtype=torch.long)}

run_scenario('IMG_CLF', lambda: RegularCNN(), im_batch, im_collate, d_model=512)

# ---- SENT_CLF（桩 BERT）----
def se_batch(X, y):
    return {'input_ids': X, 'attention_mask': torch.ones_like(X), 'labels': y}

def se_collate(batch):
    return {'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'labels': torch.tensor([b['labels'] for b in batch], dtype=torch.long)}

run_scenario('SENT_CLF', lambda: StubBertClassifier(), se_batch, se_collate, d_model=16)

shutil.rmtree(tmp, ignore_errors=True)
print('ALL SMOKE PASSED')
