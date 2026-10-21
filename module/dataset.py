import os
import torch
import csv
import json
import random
import pickle
import mat73
import math
import numpy as np
import pandas as pd
import concurrent.futures

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

CACHE_DIR = "dataset/cache"
MAX_SHARD_SIZE = 5000


def _load_image_worker(args):
    img_path, target_size = args
    try:
        img = Image.open(img_path).convert('RGB')
        if target_size is not None:
            img = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor()
            ])(img)
        return img
    except Exception:
        return None


def _img_to_uint8(img):
    """将 float32 [0,1] 图像量化为 uint8 [0,255] 存入缓存。

    ToTensor() 的输出为 k/255（k 为原始 0-255 整数），乘回 255 再四舍五入
    可无损还原原始像素，因此 uint8 与 float32 缓存数值完全等价，而内存/磁盘
    占用降为 1/4（FairFace 缓存 ~52GB -> ~13GB/进程）。
    """
    return (img * 255.0).round_().clamp_(0, 255).to(torch.uint8)


def _img_to_float(img):
    """缓存读取时还原为 float32 [0,1]；非 uint8 原样返回（向后兼容旧缓存）。"""
    if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
        return img.to(torch.float32).div_(255.0)
    return img


def _get_cache_dir(dataset_name, split):
    cache_path = os.path.join(CACHE_DIR, dataset_name, split)
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def _shard_exists(cache_dir, total_len, expected_format=None):
    if not os.path.exists(cache_dir):
        return False
    meta_path = os.path.join(cache_dir, "meta.pt")
    if not os.path.exists(meta_path):
        return False
    try:
        meta = torch.load(meta_path, weights_only=False)
        if meta.get("total_len") != total_len:
            return False
        if expected_format is not None and meta.get("format") != expected_format:
            return False
        num_shards = meta["num_shards"]
        if not isinstance(num_shards, int) or num_shards < 0:
            return False
        # 非空数据的缓存必须至少含一个分片。音频分片数按「成功解码的样本数」
        # 计，若整批音频都解码失败会写出 total_len>0 却 num_shards=0 的空缓存；
        # 这种缓存必须判为失效以触发重建，否则会以 0 样本进入分区并报
        # "client 0 has 0 samples" 这类难以定位的错误。
        if total_len > 0 and num_shards == 0:
            return False
        for index in range(num_shards):
            shard_path = os.path.join(cache_dir, f"shard_{index}.pt")
            if not os.path.exists(shard_path):
                return False
        return True
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError):
        return False


def _atomic_torch_save(value, path):
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(value, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _save_shards(cache_dir, items_list):
    total_len = len(items_list)
    num_shards = math.ceil(total_len / MAX_SHARD_SIZE)
    for i in range(num_shards):
        start = i * MAX_SHARD_SIZE
        end = min(start + MAX_SHARD_SIZE, total_len)
        shard = items_list[start:end]
        torch.save(shard, os.path.join(cache_dir, f"shard_{i}.pt"))
    meta = {"total_len": total_len, "num_shards": num_shards, "shard_size": MAX_SHARD_SIZE}
    torch.save(meta, os.path.join(cache_dir, "meta.pt"))


def _save_text_shards_stacked(cache_dir, input_ids_list, attention_mask_list, labels_list, protected_list):
    """以 stacked tensor 格式保存文本tokenization缓存（高效版）"""
    meta_path = os.path.join(cache_dir, "meta.pt")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    total_len = len(input_ids_list)
    num_shards = math.ceil(total_len / MAX_SHARD_SIZE)
    for index in range(num_shards):
        start = index * MAX_SHARD_SIZE
        end = min(start + MAX_SHARD_SIZE, total_len)
        shard = {
            'input_ids': torch.stack(input_ids_list[start:end], dim=0),
            'attention_mask': torch.stack(attention_mask_list[start:end], dim=0),
            'labels': torch.tensor(labels_list[start:end], dtype=torch.long),
            'protected': torch.tensor(protected_list[start:end], dtype=torch.long)
        }
        _atomic_torch_save(
            shard,
            os.path.join(cache_dir, f"shard_{index}.pt"),
        )
    meta = {
        "total_len": total_len,
        "num_shards": num_shards,
        "shard_size": MAX_SHARD_SIZE,
        "format": "stacked_text",
    }
    _atomic_torch_save(meta, meta_path)


def _load_shards(cache_dir):
    meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    num_shards = meta["num_shards"]
    items_list = []
    for i in range(num_shards):
        shard = torch.load(os.path.join(cache_dir, f"shard_{i}.pt"), weights_only=False)
        items_list.extend(shard)
    return items_list, meta


def _load_shards_stacked(cache_dir):
    """加载 stacked 格式的 shard（每个 shard 是一个 {img, labels, protected} 大 tensor dict）"""
    meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    num_shards = meta["num_shards"]

    # 检测 shard 格式：dict（stacked）或 list（legacy）
    first_shard = torch.load(os.path.join(cache_dir, "shard_0.pt"), weights_only=False)
    if isinstance(first_shard, list):
        # 旧格式：list of dicts
        import warnings
        warnings.warn(
            f"[Cache] Legacy cache format (list) detected in '{cache_dir}'. "
            f"Deprecated and slower. "
            f"Delete the cache dir and re-run to rebuild with stacked format. "
            f"Rebuild: shutil.rmtree('{cache_dir}')",
            DeprecationWarning,
            stacklevel=3,
        )
        print(f"[Cache] WARNING / 警告: Legacy cache format detected / 检测到旧版缓存格式 (list) "
              f"in '{cache_dir}'. "
              f"Please delete and re-run to rebuild / 请删除缓存目录后重新运行以重建新格式. "
              f"Rebuild command / 重建命令: shutil.rmtree(r'{cache_dir}')")
        items_list = [first_shard]
        for i in range(1, num_shards):
            shard = torch.load(os.path.join(cache_dir, f"shard_{i}.pt"), weights_only=False)
            items_list.append(shard)
        items_list = [item for sublist in items_list for item in sublist]
        return items_list, meta

    # 新格式：stacked dict
    all_imgs, all_labels, all_protected = [first_shard['img']], [first_shard['labels']], [first_shard['protected']]
    for i in range(1, num_shards):
        shard = torch.load(os.path.join(cache_dir, f"shard_{i}.pt"), weights_only=False)
        all_imgs.append(shard['img'])
        all_labels.append(shard['labels'])
        all_protected.append(shard['protected'])
    return {
        'img': torch.cat(all_imgs, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'protected': torch.cat(all_protected, dim=0)
    }, meta


class CachedImageDataset(Dataset):
    def __init__(self, cache_dir, lazy=False):
        """
        Args:
            cache_dir: 缓存目录
            lazy: 懒加载模式。True 时只加载 meta，按需加载单个 shard，
                  避免一次性将所有图片数据加载到内存。
        """
        self.cache_dir = cache_dir
        self.lazy = lazy
        self._shard_cache = {}  # shard_idx -> shard dict（LRU 缓存）

        if lazy:
            # 懒加载：只读 meta，不加载任何 shard 数据
            self.meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
            self._is_dict = True
            self.items = None  # 不预加载数据
        else:
            # 急加载：一次性加载所有 shard（原有行为）
            self.items, self.meta = _load_shards_stacked(cache_dir)
            self._is_dict = isinstance(self.items, dict)

    def __len__(self):
        return self.meta["total_len"]

    def __getitem__(self, index):
        if self._is_dict and self.items is not None:
            # 急加载模式：直接从预加载的 dict 中取
            return {
                'img': _img_to_float(self.items['img'][index]),
                'labels': self.items['labels'][index],
                'protected': self.items['protected'][index]
            }
        elif self._is_dict and self.items is None:
            # 懒加载模式：按需加载 shard
            shard_size = self.meta.get("shard_size", 0)
            if shard_size > 0:
                shard_idx = index // shard_size
                item_idx = index % shard_size
            else:
                # 旧格式或无 shard_size，回退到顺序加载
                shard_idx = 0
                item_idx = index

            if shard_idx not in self._shard_cache:
                shard_path = os.path.join(self.cache_dir, f"shard_{shard_idx}.pt")
                self._shard_cache[shard_idx] = torch.load(shard_path, weights_only=False)
                # LRU：最多缓存 2 个 shard
                if len(self._shard_cache) > 2:
                    oldest = next(iter(self._shard_cache))
                    del self._shard_cache[oldest]

            shard = self._shard_cache[shard_idx]
            if isinstance(shard, list):
                return shard[item_idx]
            return {
                'img': _img_to_float(shard['img'][item_idx]),
                'labels': shard['labels'][item_idx],
                'protected': shard['protected'][item_idx]
            }
        else:
            # list 格式（legacy cache）
            return self.items[index]


class CachedTextDataset(Dataset):
    def __init__(self, cache_dir):
        meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
        cache_format = meta.get("format", "legacy")
        
        if cache_format == "stacked_text":
            # 新格式：stacked tensor，直接加载所有 shard
            self._format = "stacked_text"
            self.meta = meta
            self.items = None  # dict of stacked tensors
            self._init_stacked(cache_dir, meta)
        else:
            # 旧格式：list of dicts
            self._format = "legacy"
            self.items, self.meta = _load_shards(cache_dir)
    
    def _init_stacked(self, cache_dir, meta):
        """初始化 stacked_text 格式的缓存"""
        num_shards = meta["num_shards"]
        all_input_ids, all_attn_mask, all_labels, all_protected = [], [], [], []
        for i in range(num_shards):
            shard = torch.load(os.path.join(cache_dir, f"shard_{i}.pt"), weights_only=False)
            all_input_ids.append(shard['input_ids'])
            all_attn_mask.append(shard['attention_mask'])
            all_labels.append(shard['labels'])
            all_protected.append(shard['protected'])
        self._stacked_cache = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attn_mask, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'protected': torch.cat(all_protected, dim=0)
        }
        self.items = self._stacked_cache  # 兼容旧接口

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if self._format == "stacked_text":
            return {
                'input_ids': self._stacked_cache['input_ids'][index],
                'attention_mask': self._stacked_cache['attention_mask'][index],
                'labels': self._stacked_cache['labels'][index],
                'protected': self._stacked_cache['protected'][index]
            }
        return self.items[index]


class CustomizedTabularDataset(Dataset):
    def __init__(self, attribute_dict):
        # self.raw_X = attribute_dict['raw_X']
        # self.raw_X_mask_s1 = attribute_dict['raw_X_mask_s1']
        # self.raw_X_mask_s2 = attribute_dict['raw_X_mask_s2']
        # self.raw_X_mask_s1_s2 = attribute_dict['raw_X_mask_s1_s2']

        self.s1 = np.array(attribute_dict['s1'])

        # self.s2 = np.array(attribute_dict['s2'])
        # self.X_mask_s1_s2 = np.array(attribute_dict['X_mask_s1_s2'])
        # self.X_mask_s1 = np.array(attribute_dict['X_mask_s1'])
        # self.X_mask_s2 = np.array(attribute_dict['X_mask_s2'])

        self.protected = np.array(attribute_dict['s1'])

        # self.X = torch.from_numpy(attribute_dict['X'], requires_grad=True)
        # self.X.requires_grad = True
        self.X = torch.tensor(attribute_dict['X'])
        # self.X = np.array(attribute_dict['X'])
        self.y = np.array(attribute_dict['y'])

        self.labels = np.array(attribute_dict['y'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"protected": self.s1[idx],
                # "s2": self.s2[idx],
                # "X_mask_s1_s2": self.X_mask_s1_s2[idx],
                # "X_mask_s1": self.X_mask_s1[idx],
                # "X_mask_s2": self.X_mask_s2[idx],
                "X": self.X[idx],
                "labels": self.y[idx]
                }


class CustomizedImageDataset(VisionDataset):
    def __init__(self, img_dir, img_names, labels, protected, transform, cache_name=None, cache_split=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.labels = labels
        self.protected = protected
        self.transform = transform

        if cache_name is not None and cache_split is not None:
            cache_dir = _get_cache_dir(cache_name, cache_split)
            if _shard_exists(cache_dir, len(img_names)):
                # 判断是否使用懒加载：环境变量显式指定 > 自动检测（可用内存 < 30% 空闲）。
                # 显式设为 0/false/no 时强制急加载：批量实验（多进程并发 + 训练集 shuffle）
                # 下懒加载会造成 shard LRU 抖动（每次取样本几乎都触发整块磁盘读），
                # 内存是否够用由启动器统一把关，不在此处静默降级。
                lazy_env = os.environ.get("IMAGE_CACHE_LAZY", "").strip().lower()
                if lazy_env in ("1", "true", "yes"):
                    lazy = True
                elif lazy_env in ("0", "false", "no"):
                    lazy = False
                else:
                    lazy = False
                    try:
                        import psutil
                        mem = psutil.virtual_memory()
                        if mem.available / mem.total < 0.3:
                            lazy = True
                            print(f"[Cache] Memory tight ({mem.available/1024**3:.1f}GB free / {mem.total/1024**3:.1f}GB total), "
                                  f"using lazy loading for {cache_name} {cache_split}")
                    except Exception:
                        pass
                mode_str = "lazy" if lazy else "eager"
                print(f"[Cache] Loading {cache_name} {cache_split} from cache ({mode_str}): {cache_dir}")
                cached = CachedImageDataset(cache_dir, lazy=lazy)
                self._cached_items = cached.items
                self._cache_meta = cached.meta
                self._use_cache = True
                # 懒加载时需要 cache_dir 和 shard_size 供 __getitem__ 使用
                if lazy:
                    self._lazy_cache_dir = cache_dir
                    self._lazy_shard_size = cached.meta.get("shard_size", 0)
                    self._lazy_shard_cache = cached._shard_cache
            else:
                print(f"[Cache] No valid cache found for {cache_name} {cache_split}, "
                      f"building cache before DataLoader workers start")
                self._cache_dir = cache_dir
                self._use_cache = False
                self._cache_built = False
                # 在主进程内完成构建，避免多个 DataLoader worker 并发写同一
                # shard 文件、进而读到未写完的文件（EOFError）。构建时机与
                # 文本数据集 _CachedTextClassificationDataset 保持一致。
                self._build_cache()
        else:
            self._use_cache = False
            self._cache_built = False

    def _build_cache(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        print(f"[Cache] Building image cache to: {self._cache_dir} ({len(self.img_names)} samples, {min(os.cpu_count() or 4, 4)} workers)")

        target_size = None
        if hasattr(self, 'transform') and self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    target_size = t.size
                    break

        args_list = [
            (os.path.join(self.img_dir, self.img_names[i]), target_size)
            for i in range(len(self.img_names))
        ]

        num_workers = min(os.cpu_count() or 4, 4)
        shard_idx = 0
        shard_imgs, shard_labels, shard_protected = [], [], []
        skipped = 0
        total_valid = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i, img_tensor in enumerate(executor.map(_load_image_worker, args_list, chunksize=256)):
                if img_tensor is not None:
                    shard_imgs.append(_img_to_uint8(img_tensor))
                    shard_labels.append(torch.tensor(self.labels[i], dtype=torch.float))
                    shard_protected.append(torch.tensor(self.protected[i], dtype=torch.long))
                    total_valid += 1
                else:
                    skipped += 1

                if len(shard_imgs) >= MAX_SHARD_SIZE:
                    shard = {
                        'img': torch.stack(shard_imgs, dim=0),
                        'labels': torch.stack(shard_labels, dim=0),
                        'protected': torch.stack(shard_protected, dim=0)
                    }
                    _atomic_torch_save(shard, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
                    shard_imgs, shard_labels, shard_protected = [], [], []
                    shard_idx += 1

                if (i + 1) % 5000 == 0:
                    print(f"[Cache]   Processed {i + 1}/{len(self.img_names)} (skipped {skipped})")

        if shard_imgs:
            shard = {
                'img': torch.stack(shard_imgs, dim=0),
                'labels': torch.stack(shard_labels, dim=0),
                'protected': torch.stack(shard_protected, dim=0)
            }
            _atomic_torch_save(shard, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
            shard_idx += 1

        meta = {"total_len": total_valid, "num_shards": shard_idx, "shard_size": MAX_SHARD_SIZE}
        _atomic_torch_save(meta, os.path.join(self._cache_dir, "meta.pt"))

        self._cache_meta = meta
        self._use_cache = True
        self._cache_built = True
        if skipped > 0:
            print(f"[Cache]   Total skipped {skipped} corrupted images")
        print(f"[Cache] Image cache built successfully ({total_valid} valid samples)")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # 懒加载模式：通过 CachedImageDataset 的 shard 缓存按需取数据
        if self._use_cache and hasattr(self, '_lazy_cache_dir'):
            shard_size = self._lazy_shard_size
            if shard_size > 0:
                shard_idx = index // shard_size
                item_idx = index % shard_size
            else:
                shard_idx = 0
                item_idx = index
            if shard_idx not in self._lazy_shard_cache:
                shard_path = os.path.join(self._lazy_cache_dir, f"shard_{shard_idx}.pt")
                self._lazy_shard_cache[shard_idx] = torch.load(shard_path, weights_only=False)
                if len(self._lazy_shard_cache) > 2:
                    oldest = next(iter(self._lazy_shard_cache))
                    del self._lazy_shard_cache[oldest]
            shard = self._lazy_shard_cache[shard_idx]
            if isinstance(shard, list):
                return shard[item_idx]
            return {
                'img': _img_to_float(shard['img'][item_idx]),
                'labels': shard['labels'][item_idx],
                'protected': shard['protected'][item_idx]
            }
        if self._use_cache and hasattr(self, '_cached_items') and self._cached_items is not None:
            # 新格式（stacked dict）缓存：_cached_items 为 {'img','labels','protected'} 的大 tensor
            items = self._cached_items
            if isinstance(items, dict):
                return {
                    'img': _img_to_float(items['img'][index]),
                    'labels': items['labels'][index],
                    'protected': items['protected'][index],
                }
            # 旧格式（list of dict）缓存：
            return items[index]
        if self._use_cache:
            shard_idx = index // self._cache_meta["shard_size"]
            item_idx = index % self._cache_meta["shard_size"]
            shard_path = os.path.join(self._cache_dir, f"shard_{shard_idx}.pt")
            if not hasattr(self, '_shard_cache'):
                self._shard_cache = {}
            if shard_idx not in self._shard_cache:
                self._shard_cache[shard_idx] = torch.load(shard_path, weights_only=False)
                if len(self._shard_cache) > 3:
                    oldest = next(iter(self._shard_cache))
                    del self._shard_cache[oldest]
            shard = self._shard_cache[shard_idx]
            return {
                'img': _img_to_float(shard['img'][item_idx]),
                'labels': shard['labels'][item_idx],
                'protected': shard['protected'][item_idx]
            }
        if not self._cache_built and hasattr(self, '_cache_dir'):
            self._build_cache()
            return self.__getitem__(index)
        img = Image.open(os.path.join(self.img_dir, self.img_names[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        protected_label = self.protected[index]

        return {
            'img': img,
            'labels': torch.tensor(label, dtype=torch.float),
            'protected': torch.tensor(protected_label, dtype=torch.long)
        }


class _CachedTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, protected, tokenizer, max_len,
                 cache_name=None, cache_split=None):
        self.texts = texts
        self.labels = labels
        self.protected = protected
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._use_cache = False
        self._cache_built = False

        if cache_name is None or cache_split is None:
            return

        self._cache_dir = _get_cache_dir(cache_name, cache_split)
        if _shard_exists(
            self._cache_dir,
            len(self.texts),
            expected_format="stacked_text",
        ):
            print(
                f"[Cache] Loading {cache_name} {cache_split} from cache: "
                f"{self._cache_dir}"
            )
            self._load_cache()
        else:
            print(
                f"[Cache] No valid cache found for {cache_name} {cache_split}; "
                "building cache before DataLoader workers start"
            )
            self._build_cache()

    def _load_cache(self):
        cached = CachedTextDataset(self._cache_dir)
        self._cached_items = cached.items
        self._use_cache = True
        self._cache_built = True

    def _build_cache(self):
        print(
            f"[Cache] Building text cache to: {self._cache_dir} "
            f"({len(self.texts)} samples)"
        )

        texts_batch = [str(text) for text in self.texts]
        batch_size = 256
        input_ids_batches = []
        attention_mask_batches = []

        for start in range(0, len(texts_batch), batch_size):
            end = min(start + batch_size, len(texts_batch))
            encoding = self.tokenizer(
                texts_batch[start:end],
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids_batches.append(encoding['input_ids'])
            attention_mask_batches.append(encoding['attention_mask'])
            if end % 50000 == 0:
                print(f"[Cache]   Tokenized {end}/{len(self.texts)}")

        all_input_ids = torch.cat(input_ids_batches, dim=0)
        all_attention_masks = torch.cat(attention_mask_batches, dim=0)
        _save_text_shards_stacked(
            self._cache_dir,
            list(all_input_ids),
            list(all_attention_masks),
            list(self.labels),
            list(self.protected),
        )
        self._load_cache()
        print("[Cache] Text cache built successfully (batch mode)")

    def _cached_item(self, item):
        if isinstance(self._cached_items, dict):
            return {
                'input_ids': self._cached_items['input_ids'][item],
                'attention_mask': self._cached_items['attention_mask'][item],
                'labels': self._cached_items['labels'][item],
                'protected': self._cached_items['protected'][item],
            }
        return self._cached_items[item]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self._use_cache:
            return self._cached_item(item)

        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long),
            'protected': torch.tensor(self.protected[item], dtype=torch.long),
        }


# MoJi 情感分析二分类数据集 / MoJi sentiment-analysis (binary) dataset
# 来源 / Source: HuggingFace LabHC/moji (默认 80% 置信度 revision)，基于 Blodgett et al.,
#   EMNLP 2016 (D16-1120) 的 TwitterAAE 语料: http://slanglab.cs.umass.edu/TwitterAAE/
# 列语义 / Column semantics (映射见 experiment_setup.py 的 moji 分支):
#   text   -> 原始推文（实体已匿名化），tokenize 后缓存为 input_ids / raw tweet
#   label  -> 情感极性 / sentiment polarity: 0=消极(negative), 1=积极(positive)
#   sa     -> 敏感属性 = 推文所用英语变体（方言模型预测 + 置信度筛选）:
#             0=AAE(非裔美式/黑人英语), 1=SAE(标准美式英语)
#             sensitive attribute = English dialect: 0=African-American English, 1=Standard American English
# 训练集1613790条(AAE 84266) / train: 1,613,790 (AAE 84,266)
# 测试集448276条(AAE 23407) / test: 448,276 (AAE 23,407)
class MoJiDataset(_CachedTextClassificationDataset):
    pass


# 训练集59179条记录
# 测试集12682条记录
class MTCDataset(Dataset):
    def __init__(self, input_ids, labels, masks, protected):
        self.input_ids = torch.tensor(np.array(input_ids))
        self.masks = torch.tensor(masks)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.protected = torch.tensor(protected, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        input_id = self.input_ids[item]
        mask = self.masks[item]
        label = self.labels[item]
        protected_label = self.protected[item]

        return {
            'input_ids': input_id,
            'attention_mask': mask,
            'labels': label,
            'protected': protected_label
        }


# Bios 偏见二分类数据集 / Bios bias binary-classification dataset
# 列语义 / Column semantics (映射见 experiment_setup.py 的 bios 分支):
#   text   -> 原始简历文本 hard_text / raw bio text
#   label  -> 职业 profession(0-27) 经 bios_binary 归并: {2,18,19,21}->1, 其余->0
#   gender -> 敏感属性/group 列：男=0，女=1 / sensitive attribute: male=0, female=1
# 训练集257478条 / train: 257,478 samples
# 测试集99069条 / test: 99,069 samples
class BiosDataset(_CachedTextClassificationDataset):
    pass

# 总共202599张图片
# 训练集162770条记录
# 验证集19867条记录
# 测试集19962条记录
class CelebaDataset(VisionDataset):
    """Custom Dataset for loading CelebA face images"""
    # 40个二元属性，从1开始。其中：第3号属性为吸引人的（Attractive），第21号属性为男性（Male）。

    def __init__(self, data_dir=r'dataset/celeba', split='train', image_size=(64, 64)):

        rep_file = os.path.join(data_dir, 'Eval/list_eval_partition.txt')
        self.img_dir = os.path.join(data_dir, 'Img/img_align_celeba/')
        self.ann_file = os.path.join(data_dir, 'Anno/list_attr_celeba.txt')
        self.image_size = image_size

        with open(rep_file) as f:
            rep = f.read()
        rep = [elt.split() for elt in rep.split('\n')]
        rep.pop()

        with open(self.ann_file, 'r') as f:
            data = f.read() # ann_file 是-1和1两种取值
        data = data.split('\n')
        names = data[1].split() # data[0]表示第0行，数字202599表示数据集大小; data[1]表示第1行，记录了各种属性的名字
        data = [elt.split() for elt in data[2:]] # 第2行开始就是名字+属性的记录
        data.pop()

        self.img_names = []
        self.labels = []
        self.protected = []
        for k in range(len(data)):
            assert data[k][0] == rep[k][0]
            if (split == 'train' and int(rep[k][1]) == 0) or \
                    (split == 'val' and int(rep[k][1]) == 1) or \
                    (split == 'test' and int(rep[k][1]) == 2):
                self.img_names.append(data[k][0])
                # self.labels.append([1 if elt == '1' else 0 for elt in data[k][1:]])
                self.labels.append(1 if data[k][3] == '1' else 0) # 我们以3号属性作为分类标签，21号属性作为敏感属性。
                self.protected.append(1 if data[k][21] == '1' else 0) # 我们以3号属性作为分类标签，21号属性作为敏感属性。

        target_size = image_size
        self.transform = [transforms.Resize(target_size), transforms.ToTensor()]
        self.transform = transforms.Compose(self.transform)
        self.labels_rep = [[i] for i in range(40)]

        cache_dir = _get_cache_dir("celeba", split)
        if _shard_exists(cache_dir, len(self.img_names)):
            print(f"[Cache] Loading CelebA {split} from cache: {cache_dir}")
            cached = CachedImageDataset(cache_dir)
            self._cached_items = cached.items
            self._cache_meta = cached.meta
            self._use_cache = True
        else:
            print(f"[Cache] No valid cache found for CelebA {split}, "
                  f"building cache before DataLoader workers start")
            self._cache_dir = cache_dir
            self._use_cache = False
            self._cache_built = False
            # 同 CustomizedImageDataset：在主进程内构建，避免 worker 并发写
            # 同一 shard 文件导致的读取不完整（EOFError）。
            self._build_cache()

    def _build_cache(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        print(f"[Cache] Building CelebA cache to: {self._cache_dir} ({len(self.img_names)} samples, {min(os.cpu_count() or 4, 4)} workers)")

        args_list = [
            (os.path.join(self.img_dir, self.img_names[i]), self.image_size)
            for i in range(len(self.img_names))
        ]

        num_workers = min(os.cpu_count() or 4, 4)
        shard_idx = 0
        shard_imgs, shard_labels, shard_protected = [], [], []
        skipped = 0
        total_valid = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i, img_tensor in enumerate(executor.map(_load_image_worker, args_list, chunksize=256)):
                if img_tensor is not None:
                    shard_imgs.append(_img_to_uint8(img_tensor))
                    shard_labels.append(torch.tensor(self.labels[i], dtype=torch.float))
                    shard_protected.append(torch.tensor(self.protected[i], dtype=torch.long))
                    total_valid += 1
                else:
                    skipped += 1

                if len(shard_imgs) >= MAX_SHARD_SIZE:
                    shard = {
                        'img': torch.stack(shard_imgs, dim=0),
                        'labels': torch.stack(shard_labels, dim=0),
                        'protected': torch.stack(shard_protected, dim=0)
                    }
                    _atomic_torch_save(shard, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
                    shard_imgs, shard_labels, shard_protected = [], [], []
                    shard_idx += 1

                if (i + 1) % 10000 == 0:
                    print(f"[Cache]   Processed {i + 1}/{len(self.img_names)} (skipped {skipped})")

        if shard_imgs:
            shard = {
                'img': torch.stack(shard_imgs, dim=0),
                'labels': torch.stack(shard_labels, dim=0),
                'protected': torch.stack(shard_protected, dim=0)
            }
            _atomic_torch_save(shard, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
            shard_idx += 1

        meta = {"total_len": total_valid, "num_shards": shard_idx, "shard_size": MAX_SHARD_SIZE}
        _atomic_torch_save(meta, os.path.join(self._cache_dir, "meta.pt"))

        self._cache_meta = meta
        self._use_cache = True
        self._cache_built = True
        if skipped > 0:
            print(f"[Cache]   Total skipped {skipped} corrupted images")
        print(f"[Cache] CelebA cache built successfully ({total_valid} valid samples)")

    def __getitem__(self, index):
        if self._use_cache and hasattr(self, '_cached_items'):
            # 新格式（stacked dict）缓存：_cached_items 为 {'img','labels','protected'} 的大 tensor
            items = self._cached_items
            if isinstance(items, dict):
                return {
                    'img': _img_to_float(items['img'][index]),
                    'labels': items['labels'][index],
                    'protected': items['protected'][index],
                }
            # 旧格式（list of dict）缓存：
            return items[index]
        if not self._use_cache and not self._cache_built:
            self._build_cache()
        if self._use_cache or self._cache_built:
            shard_idx = index // self._cache_meta["shard_size"]
            item_idx = index % self._cache_meta["shard_size"]
            shard_path = os.path.join(self._cache_dir, f"shard_{shard_idx}.pt")
            if not hasattr(self, '_shard_cache'):
                self._shard_cache = {}
            if shard_idx not in self._shard_cache:
                self._shard_cache[shard_idx] = torch.load(shard_path, weights_only=False)
                if len(self._shard_cache) > 3:
                    oldest = next(iter(self._shard_cache))
                    del self._shard_cache[oldest]
            shard = self._shard_cache[shard_idx]
            return {
                'img': _img_to_float(shard['img'][item_idx]),
                'labels': shard['labels'][item_idx],
                'protected': shard['protected'][item_idx]
            }
        img = Image.open(os.path.join(self.img_dir, self.img_names[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {
            'img': img,
            'labels': torch.tensor(self.labels[index], dtype=torch.float),
            'protected': torch.tensor(self.protected[index], dtype=torch.long)
        }

    def __len__(self):
        if self._use_cache or self._cache_built:
            return self._cache_meta["total_len"]
        return len(self.img_names)



# 总共24108张图片
def get_UTKFace_dataset(img_dir=r'./dataset/UTKFace/img', image_size=(64, 64)):
    # 图像名称中嵌入了每张人脸图像的标签，格式为[age]_[gender]_[race]_[date&time].jpg
    # 1、[age]：是 0 到 116 之间的整数，表示年龄；
    # 2、[gender]：是 0（男性）或 1（女性）；
    # 3、[race]： 是 0 到 4 之间的整数，表示白人、黑人、亚洲人、印度人和其他人（如西班牙裔、拉丁裔、中东裔）；
    # 4、[date&time]：格式为 yyyymmddHHMMSSFFF，显示图像收集到 UTKFace 的日期和时间。

    target_size = image_size
    transform = [transforms.Resize(target_size), transforms.ToTensor()]
    transform = transforms.Compose(transform)

    img_names = os.listdir(img_dir)
    labels = []
    protected = []

    for name in img_names:
        if ".jpg" in name:
            tmp = name.split("_")
            gender = tmp[1] # 0（男性）或 1（女性）
            race = tmp[2] # 0 到 4 之间的整数，表示白人、黑人、亚洲人、印度人和其他人（如西班牙裔、拉丁裔、中东裔）
            # 男性：1，女性：0；
            labels.append(1 if gender == '0' else 0)  # 以性别属性作为分类标签。
            # 黑人0，其他人种：1
            protected.append(0 if race == '1' else 1)  # 以种族属性作为敏感属性。

    img_count = len(labels)

    if os.path.exists(r'./dataset/UTKFace/training_indexes.pickle'):
        with open(r'./dataset/UTKFace/training_indexes.pickle', "rb") as f:
            training_indexes = pickle.load(f)
    else:
        training_size = int(img_count * 0.8)
        random.seed(42)
        training_indexes = random.sample(range(0, img_count), training_size)
        with open(r'./dataset/UTKFace/training_indexes.pickle', "wb") as f:
            pickle.dump(training_indexes, f)

    training_img_names, training_labels, training_protected = [], [], []
    test_img_names, test_labels, test_protected = [], [], []
    for i, item in enumerate(labels):
        if i in training_indexes:
            training_img_names.append(img_names[i])
            training_labels.append(item)
            training_protected.append(protected[i])
        else:
            test_img_names.append(img_names[i])
            test_labels.append(item)
            test_protected.append(protected[i])

    # Constructing the training dataset
    training_dataset = CustomizedImageDataset(img_dir, training_img_names, training_labels, training_protected, transform, cache_name="UTKFace", cache_split="train")
    # Constructing the testing dataset
    testing_dataset = CustomizedImageDataset(img_dir, test_img_names, test_labels, test_protected, transform, cache_name="UTKFace", cache_split="test")

    return training_dataset, testing_dataset


# 总共86744 + 10954 张图片
def get_FairFace_dataset(img_dir=r'./dataset/FairFace/', image_size=(224, 224)):
    # 路径、年龄段、性别、人种、分类
    '''
    image: The image
    age: Age class among ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
    gender: Gender class among ["Male", "Female"]
    race: Race class among ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
    service_test: Not sure what this is. See issue.
    '''

    target_size = image_size
    transform = [transforms.Resize(target_size), transforms.ToTensor()]
    transform = transforms.Compose(transform)

    training_img_names = os.listdir(img_dir+"train")
    training_labels, training_protected = [], []
    df = pd.read_csv(r'./dataset/FairFace/fairface_label_train.csv', encoding='utf-8')
    df_array = np.array(df)
    df_list = df_array.tolist()
    for item in df_list:
        gender = item[2]
        training_protected.append(1 if gender == 'Male' else 0)
        service_test = item[4]
        training_labels.append(1 if service_test else 0)

    test_img_names = os.listdir(img_dir + "val")
    test_labels, test_protected = [], []
    df = pd.read_csv(r'./dataset/FairFace/fairface_label_val.csv', encoding='utf-8')
    df_array = np.array(df)
    df_list = df_array.tolist()
    for item in df_list:
        gender = item[2]
        test_protected.append(1 if gender == 'Male' else 0)
        service_test = item[4]
        test_labels.append(1 if service_test else 0)

    # Constructing the training dataset
    training_dataset = CustomizedImageDataset(img_dir+"fairface-img-margin025-trainval/train", training_img_names, training_labels, training_protected, transform, cache_name="FairFace", cache_split="train")
    # Constructing the testing dataset
    testing_dataset = CustomizedImageDataset(img_dir + "fairface-img-margin025-trainval/val", test_img_names, test_labels, test_protected, transform, cache_name="FairFace", cache_split="test")

    return training_dataset, testing_dataset


# 总共19284 + 4821 张图片
def get_LFWAPlus_dataset(img_dir=r'./dataset/LFWA+/', image_size=(250, 250), max_samples=None):
    # 40个二元属性，从0开始。其中：第02号属性为吸引人的（Attractive），第20号属性为男性（Male）。

    target_size = image_size
    transform = [transforms.Resize(target_size), transforms.ToTensor()]
    transform = transforms.Compose(transform)


    path_mat = img_dir+"lfw_att_40.mat"
    mat = mat73.loadmat(path_mat)
    # attr: list 40
    # label: ndarray(13143,40)
    # name: list 13143
    mat_attr, mat_label, mat_name = mat['AttrName'], mat['label'], mat['name']
    img_names = [img_dir + 'lfw/' + item.replace('\\','/') for item in mat_name]
    labels = []
    protected = []

    for index, img in enumerate(img_names):
        gender = mat_label[index][20] # 1（男性）或 0（女性）
        attractive = mat_label[index][2] # 1 是 或 0 否
        protected.append(1 if (gender == 1) or (str(gender) == '1.0') else 0)  # 以性别属性作为敏感属性。
        labels.append(1 if (attractive == 1) or (str(attractive) == '1.0') else 0)

    img_count = len(labels)


    if os.path.exists(r'./dataset/LFWA+/training_indexes.pickle'):
        with open(r'./dataset/LFWA+/training_indexes.pickle', "rb") as f:
            training_indexes = pickle.load(f)
    else:
        training_size = int(img_count * 0.8)
        random.seed(42)
        training_indexes = random.sample(range(0, img_count), training_size)
        with open(r'./dataset/LFWA+/training_indexes.pickle', "wb") as f:
            pickle.dump(training_indexes, f)

    training_img_names, training_labels, training_protected = [], [], []
    test_img_names, test_labels, test_protected = [], [], []
    for i, item in enumerate(labels):
        if i in training_indexes:
            training_img_names.append(img_names[i])
            training_labels.append(item)
            training_protected.append(protected[i])
        else:
            test_img_names.append(img_names[i])
            test_labels.append(item)
            test_protected.append(protected[i])

    if max_samples is not None:
        training_img_names = training_img_names[:max_samples]
        training_labels = training_labels[:max_samples]
        training_protected = training_protected[:max_samples]
        test_img_names = test_img_names[:max_samples]
        test_labels = test_labels[:max_samples]
        test_protected = test_protected[:max_samples]

    # Constructing the training dataset
    training_dataset = CustomizedImageDataset('', training_img_names, training_labels, training_protected, transform, cache_name="LFWAPlus", cache_split="train")
    # Constructing the testing dataset
    testing_dataset = CustomizedImageDataset('', test_img_names, test_labels, test_protected, transform, cache_name="LFWAPlus", cache_split="test")

    return training_dataset, testing_dataset


# ==================== 语音数据集 / Audio datasets (AUD_CLF) ====================
# 前端统一为 log-Mel 频谱：16 kHz 单声道、25 ms 窗（n_fft=400）、10 ms 帧移
# （hop_length=160）、64 个梅尔频带。帧数由各数据集按固定时长对齐（短则补静音、
# 长则截断），使同一数据集内所有样本形状一致，可直接堆成 batch。
# 前端只依赖 torchaudio（仓库既有依赖），计算一次后落盘缓存，训练阶段只读缓存。
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 64
AUDIO_N_FFT = 400
AUDIO_HOP_LENGTH = 160
AUDIO_TOP_DB = 80.0
# CREMA-D 片段普遍 2-3 秒，取 3 秒；AudioMNIST 固定 1 秒；
# RAVDESS 语音片段约 3-5 秒，同样取 3 秒与 CREMA-D 对齐
AUDIO_CREMAD_FRAMES = 300
AUDIO_AUDIOMNIST_FRAMES = 100
AUDIO_RAVDESS_FRAMES = 300


def _read_audio_mono(path, sample_rate):
    """读取音频文件为单声道 ``[1, N]`` 张量，并重采样到 ``sample_rate``。

    解码用 ``soundfile``（libsndfile）而不是 ``torchaudio.load``：torchaudio 2.9
    起移除了内置解码后端，``torchaudio.load`` 改为依赖 TorchCodec（需 FFmpeg
    动态库）。CREMA-D / AudioMNIST 均为 WAV，libsndfile 原生支持，可避免额外的
    编解码依赖，保证开箱即用。重采样复用 ``torchaudio.functional.resample``
    （纯张量实现，不经过任何编解码后端）。
    """
    import soundfile as sf

    data, original_rate = sf.read(path, dtype='float32', always_2d=True)  # [N, C]
    waveform = torch.from_numpy(data.T).contiguous()  # [N, C] -> [C, N]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform[:1]
    if original_rate != sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
    return waveform


def _load_audio_feature_worker(args):
    """把单个 wav 转成固定长度的归一化 log-Mel 频谱 ``[T, F]``。

    返回 ``None`` 表示该文件无法解码（由调用方剔除并告警）。
    """
    path, target_frames, sample_rate, n_mels, n_fft, hop_length, top_db = args
    try:
        import torchaudio

        waveform = _read_audio_mono(path, sample_rate)

        target_samples = target_frames * hop_length
        if waveform.shape[1] < target_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_samples - waveform.shape[1])
            )
        else:
            waveform = waveform[:, :target_samples]

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )(waveform)
        log_mel = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)(mel)
        feature = log_mel.squeeze(0).transpose(0, 1).contiguous()  # [F, T] -> [T, F]

        # 时间维对齐到固定帧数：短音频补静音（dB 下界），长音频截断
        if feature.shape[0] < target_frames:
            padding = feature.new_full(
                (target_frames - feature.shape[0], feature.shape[1]), float(-top_db)
            )
            feature = torch.cat([feature, padding], dim=0)
        else:
            feature = feature[:target_frames]

        # 固定量程归一化：dB 值域 [峰值 - top_db, 峰值] 线性映射到 [0, 1] 附近
        feature = (feature + top_db) / top_db
        return feature
    except Exception:
        return None


def _extract_audio_features(paths, feature_config, total_paths):
    """并行提取全部音频特征，返回 ``(特征列表, 有效样本下标)``。"""
    workers = min(os.cpu_count() or 4, 8)
    args_list = [(path,) + tuple(feature_config) for path in paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_load_audio_feature_worker, args_list))

    valid_indices = [index for index, item in enumerate(results) if item is not None]
    if len(valid_indices) != total_paths:
        print(f"[Cache] WARNING: {total_paths - len(valid_indices)} / {total_paths} "
              f"audio clips failed to decode and were dropped")
    return [results[index] for index in valid_indices], valid_indices


def _save_audio_shards(cache_dir, features, valid_indices, total_paths):
    """保存音频特征缓存（stacked float16 分片）。

    与图像缓存一致地分片；额外记录有效样本下标，使个别无法解码的音频被丢弃后，
    ``labels`` / ``protected`` 仍能与特征严格对齐。
    """
    total_valid = len(features)
    num_shards = math.ceil(total_valid / MAX_SHARD_SIZE) if total_valid else 0
    for index in range(num_shards):
        start = index * MAX_SHARD_SIZE
        end = min(start + MAX_SHARD_SIZE, total_valid)
        shard = {
            'audio': torch.stack(features[start:end], dim=0).to(torch.float16),
            'valid_indices': torch.tensor(valid_indices[start:end], dtype=torch.long),
        }
        _atomic_torch_save(shard, os.path.join(cache_dir, f"shard_{index}.pt"))
    meta = {
        'total_len': total_paths,
        'total_valid': total_valid,
        'num_shards': num_shards,
        'shard_size': MAX_SHARD_SIZE,
        'format': 'stacked_audio',
    }
    _atomic_torch_save(meta, os.path.join(cache_dir, "meta.pt"))


def _load_audio_shards(cache_dir):
    """加载音频特征缓存，返回 ``(stacked 特征, 有效样本下标)``。"""
    meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    audios, valid_indices = [], []
    for index in range(meta["num_shards"]):
        shard = torch.load(os.path.join(cache_dir, f"shard_{index}.pt"), weights_only=False)
        audios.append(shard['audio'])
        valid_indices.append(shard['valid_indices'])
    if not audios:
        return torch.empty(0), []
    return torch.cat(audios, dim=0), torch.cat(valid_indices, dim=0).tolist()


class CustomizedAudioDataset(Dataset):
    """AUD_CLF 的语音数据集：wav -> 缓存的 log-Mel 特征。

    ``__getitem__`` 返回 ``{'img': FloatTensor[T, F], 'labels': int, 'protected': int}``。
    与图像/表格数据集一致地暴露 ``labels`` / ``protected`` / ``sample_ids``，
    以满足联邦分区（``module/partition.py``）对稳定样本标识的要求。

    输入张量沿用仓库通用的键名 ``img``（即「共享基座 + 二分类头」模型的输入张量）：
    语音与图像在本仓库共享同一条处理路径，40+ 算法的 IMG_CLF 分支可直接复用于
    AUD_CLF，无需为语音任务改动各算法的取值代码。

    ``__getitem__`` returns ``{'img': FloatTensor[T, F], 'labels': int, 'protected': int}``.
    The input tensor keeps the repo-wide generic key ``img`` (the input tensor of every
    "shared base + binary head" model) so that audio reuses the IMG_CLF code path shared
    by 40+ algorithms without touching their per-batch lookups.
    """

    def __init__(self, audio_dir, audio_names, labels, protected, sample_ids,
                 target_frames, sample_rate, n_mels, n_fft, hop_length, top_db,
                 cache_name=None, cache_split=None, speaker_ids=None):
        self.audio_dir = audio_dir
        self.audio_names = list(audio_names)
        self.sample_ids = list(sample_ids)
        # 说话人标识：``Speaker`` 划分策略按整组说话人分配客户端。
        # Speaker identities: the ``Speaker`` split assigns whole speakers to clients.
        self.speaker_ids = (
            list(speaker_ids) if speaker_ids is not None else list(self.sample_ids)
        )

        feature_config = (target_frames, sample_rate, n_mels, n_fft, hop_length, top_db)
        paths = [os.path.join(audio_dir, name) for name in self.audio_names]

        if cache_name is not None and cache_split is not None:
            cache_dir = _get_cache_dir(
                f"{cache_name}_mel{n_mels}x{target_frames}", cache_split
            )
            if _shard_exists(cache_dir, len(paths), expected_format="stacked_audio"):
                print(f"[Cache] Loading audio features from cache: {cache_dir}")
                stacked, valid_indices = _load_audio_shards(cache_dir)
            else:
                print(f"[Cache] Building audio feature cache to: {cache_dir} "
                      f"({len(paths)} clips, mel {n_mels} x {target_frames})")
                features, valid_indices = _extract_audio_features(
                    paths, feature_config, len(paths)
                )
                os.makedirs(cache_dir, exist_ok=True)
                _save_audio_shards(cache_dir, features, valid_indices, len(paths))
                stacked = (torch.stack(features, dim=0).to(torch.float16)
                           if features else torch.empty(0))
        else:
            features, valid_indices = _extract_audio_features(
                paths, feature_config, len(paths)
            )
            stacked = (torch.stack(features, dim=0).to(torch.float16)
                       if features else torch.empty(0))

        # 剔除解码失败的样本，保持特征与标签/敏感属性严格对齐
        if len(valid_indices) != len(self.audio_names):
            self.audio_names = [self.audio_names[i] for i in valid_indices]
            self.sample_ids = [self.sample_ids[i] for i in valid_indices]
            self.speaker_ids = [self.speaker_ids[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            protected = [protected[i] for i in valid_indices]

        self._stacked = stacked
        self.labels = list(labels)
        self.protected = list(protected)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'img': self._stacked[index].to(torch.float32),
            'labels': self.labels[index],
            'protected': self.protected[index],
        }


def _stratified_speaker_split(speaker_groups, test_ratio, seed):
    """按群组分层挑选测试说话人，返回 ``(训练说话人, 测试说话人)``。

    按说话人（而非样本）划分可避免同一说话人的录音同时出现在训练与测试集
    造成的泄漏；按敏感属性分层可保持测试集内各群组的样本比例，避免小群组
    在随机划分下完全缺席。
    """
    by_group = {}
    for speaker, group in speaker_groups.items():
        by_group.setdefault(group, []).append(speaker)

    rng = random.Random(seed)
    test_speakers = set()
    for group in sorted(by_group, key=lambda item: str(item)):
        ordered = sorted(by_group[group])
        count = max(1, int(round(len(ordered) * test_ratio)))
        count = min(count, max(len(ordered) - 1, 0))
        test_speakers.update(rng.sample(ordered, count))

    train_speakers = sorted(s for s in speaker_groups if s not in test_speakers)
    return train_speakers, sorted(test_speakers)


def _load_cremad_demographics(demographics_csv):
    """读取 CREMA-D 说话人人口学表（官方 ``VideoDemographics.csv``）。"""
    actors = {}
    with open(demographics_csv, 'r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            actors[str(row['ActorID']).strip()] = {
                'age': int(row['Age']),
                'sex': row['Sex'].strip(),
                'race': row['Race'].strip(),
                'ethnicity': row['Ethnicity'].strip(),
            }
    return actors


def _load_audiomnist_speaker_meta(speaker_meta):
    """读取 AudioMNIST 说话人元数据（官方 ``audioMNIST_meta.txt``）。"""
    with open(speaker_meta, 'r', encoding='utf-8') as handle:
        raw = json.load(handle)
    speakers = {}
    for speaker, info in raw.items():
        speakers[str(speaker).zfill(2)] = {
            'gender': str(info.get('gender', '')).strip().lower(),
            'accent': str(info.get('accent', '')).strip(),
            'age': info.get('age'),
        }
    return speakers


def _standardize_audio_features(training_dataset, testing_dataset):
    """Fit mel-feature z-score statistics on the training split and apply to both.

    为什么必须做：Conv-RNN 主干在只经过 min-max 量级缩放的 log-Mel 特征上会坍缩为
    **与输入无关的常数表征**——初始化时每条音频都映射到几乎同一个向量（样本间
    逐维 std ≈ 0.006，而单样本向量范数 ≈ 1.03），分类头因此只能拟合 bias，模型
    只输出类先验（表现为 ACC 恒等于类先验/随机、DEO=0、FR=1，所有算法一致）。
    统计量只在训练划分上拟合，测试划分沿用训练统计量，避免信息泄漏。
    """
    stacked = training_dataset._stacked.float()
    mean = float(stacked.mean())
    std = float(stacked.std())
    for dataset in (training_dataset, testing_dataset):
        dtype = dataset._stacked.dtype
        dataset._stacked = ((dataset._stacked.float() - mean) / std).to(dtype)
    return mean, std


def get_CREMAD_dataset(audio_dir=r'./dataset/CREMA-D',
                       demographics_csv=r'./dataset/CREMA-D/crema_demographics.csv',
                       test_ratio=0.2, seed=42,
                       target_frames=AUDIO_CREMAD_FRAMES,
                       sample_rate=AUDIO_SAMPLE_RATE, n_mels=AUDIO_N_MELS,
                       n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH,
                       top_db=AUDIO_TOP_DB,
                       sensitive_attribute='race'):
    """CREMA-D 情感语音数据集（AUD_CLF）。

    - 标签：情感效价二分类，``HAP``（积极）= 1，其余 5 类（ANG/DIS/FEA/NEU/SAD）= 0；
    - 敏感属性：``sensitive_attribute='race'``（默认）时 ``Caucasian`` = 1、
      非 Caucasian = 0；``'gender'`` 时 ``Male`` = 1、``Female`` = 0；
    - 划分：按 ``ActorID``（说话人）划分，并按所选敏感属性分层；
      ``Race`` 为 ``Unknown`` 的演员（1 位）被整体丢弃；

    文件命名规则为 ``{ActorID}_{Sentence}_{Emotion}_{Intensity}.wav``，共 7442 条音频、
    91 位演员。

    为什么提供两个敏感属性 / Why two sensitive attributes
    ----------------------------------------------------
    种族分组为 61 : 29（Caucasian : 非 Caucasian），性别分组为 48 : 43，后者更均衡，
    DEO/HM 的估计方差更小。主协议仍用 race（与已有结果、README 保持一致），gender
    作为并排报告的补充口径。
    """
    if sensitive_attribute not in ('race', 'gender'):
        raise ValueError(
            f"unsupported CREMA-D sensitive_attribute: {sensitive_attribute} "
            f"(expected 'race' or 'gender')"
        )

    wav_dir = os.path.join(audio_dir, 'AudioWAV')
    audio_names = sorted(name for name in os.listdir(wav_dir)
                         if name.lower().endswith('.wav'))
    actors = _load_cremad_demographics(demographics_csv)

    labels, protected, speaker_ids = [], [], []
    speaker_groups = {}
    kept_names, kept_ids = [], []
    dropped = 0
    for name in audio_names:
        parts = os.path.splitext(name)[0].split('_')
        if len(parts) < 4:
            dropped += 1
            continue
        actor_id, emotion = parts[0], parts[2]
        info = actors.get(actor_id)
        if info is None or info['race'] == 'Unknown':
            dropped += 1
            continue

        if sensitive_attribute == 'race':
            is_protected = info['race'] == 'Caucasian'
            group = 'Caucasian' if is_protected else 'non-Caucasian'
        else:
            is_protected = info['sex'] == 'Male'
            group = 'Male' if is_protected else 'Female'

        kept_names.append(name)
        kept_ids.append(name)
        labels.append(1 if emotion == 'HAP' else 0)
        protected.append(1 if is_protected else 0)
        speaker_ids.append(actor_id)
        speaker_groups[actor_id] = group

    if dropped:
        print(f"[CREMA-D] Dropped {dropped} clips (unknown/unlisted actors)")

    train_speakers, test_speakers = _stratified_speaker_split(
        speaker_groups, test_ratio, seed
    )
    print(f"[CREMA-D] Speaker-level split: {len(train_speakers)} train / "
          f"{len(test_speakers)} test speakers")

    feature_kwargs = dict(target_frames=target_frames, sample_rate=sample_rate,
                          n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                          top_db=top_db)
    train_indexes = [i for i, speaker in enumerate(speaker_ids)
                     if speaker in set(train_speakers)]
    test_indexes = [i for i, speaker in enumerate(speaker_ids)
                    if speaker in set(test_speakers)]

    training_dataset = CustomizedAudioDataset(
        wav_dir, [kept_names[i] for i in train_indexes],
        [labels[i] for i in train_indexes], [protected[i] for i in train_indexes],
        [kept_ids[i] for i in train_indexes],
        cache_name="CREMAD", cache_split="train",
        speaker_ids=[speaker_ids[i] for i in train_indexes], **feature_kwargs)
    testing_dataset = CustomizedAudioDataset(
        wav_dir, [kept_names[i] for i in test_indexes],
        [labels[i] for i in test_indexes], [protected[i] for i in test_indexes],
        [kept_ids[i] for i in test_indexes],
        cache_name="CREMAD", cache_split="test",
        speaker_ids=[speaker_ids[i] for i in test_indexes], **feature_kwargs)

    _standardize_audio_features(training_dataset, testing_dataset)

    return training_dataset, testing_dataset


def get_AudioMNIST_dataset(audio_dir=r'./dataset/AudioMNIST/data',
                           speaker_meta=r'./dataset/AudioMNIST/audiomnist_speaker_meta.json',
                           test_ratio=0.2, seed=42,
                           target_frames=AUDIO_AUDIOMNIST_FRAMES,
                           sample_rate=AUDIO_SAMPLE_RATE, n_mels=AUDIO_N_MELS,
                           n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH,
                           top_db=AUDIO_TOP_DB):
    """AudioMNIST 数字语音数据集（AUD_CLF）。

    - 标签：数字奇偶二分类，奇数 = 1，偶数 = 0；
    - 敏感属性：性别，``male`` = 1，``female`` = 0；
    - 划分：按说话人划分，并按性别分层。

    文件命名规则为 ``{digit}_{speaker}_{index}.wav``，共 30000 条音频、60 位说话人。
    """
    speakers_info = _load_audiomnist_speaker_meta(speaker_meta)

    relative_paths = []
    for root, _dirs, files in os.walk(audio_dir):
        for file_name in files:
            if file_name.lower().endswith('.wav'):
                relative_paths.append(
                    os.path.relpath(os.path.join(root, file_name), audio_dir)
                )
    relative_paths.sort()

    labels, protected, speaker_ids = [], [], []
    speaker_groups = {}
    kept_names, kept_ids = [], []
    dropped = 0
    for relative_path in relative_paths:
        parts = os.path.splitext(os.path.basename(relative_path))[0].split('_')
        if len(parts) >= 2:
            digit_token, speaker_token = parts[0], parts[1]
        else:
            dropped += 1
            continue
        try:
            digit = int(digit_token)
            speaker = str(int(speaker_token)).zfill(2)
        except ValueError:
            dropped += 1
            continue

        info = speakers_info.get(speaker)
        if info is None or info['gender'] not in ('male', 'female'):
            dropped += 1
            continue

        kept_names.append(relative_path)
        kept_ids.append(relative_path)
        labels.append(digit % 2)
        protected.append(1 if info['gender'] == 'male' else 0)
        speaker_ids.append(speaker)
        speaker_groups[speaker] = info['gender']

    if dropped:
        print(f"[AudioMNIST] Dropped {dropped} clips (unparsable name/unknown speaker)")

    train_speakers, test_speakers = _stratified_speaker_split(
        speaker_groups, test_ratio, seed
    )
    print(f"[AudioMNIST] Speaker-level split: {len(train_speakers)} train / "
          f"{len(test_speakers)} test speakers")

    feature_kwargs = dict(target_frames=target_frames, sample_rate=sample_rate,
                          n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                          top_db=top_db)
    train_set, test_set = set(train_speakers), set(test_speakers)
    train_indexes = [i for i, speaker in enumerate(speaker_ids) if speaker in train_set]
    test_indexes = [i for i, speaker in enumerate(speaker_ids) if speaker in test_set]

    training_dataset = CustomizedAudioDataset(
        audio_dir, [kept_names[i] for i in train_indexes],
        [labels[i] for i in train_indexes], [protected[i] for i in train_indexes],
        [kept_ids[i] for i in train_indexes],
        cache_name="AudioMNIST", cache_split="train", **feature_kwargs)
    testing_dataset = CustomizedAudioDataset(
        audio_dir, [kept_names[i] for i in test_indexes],
        [labels[i] for i in test_indexes], [protected[i] for i in test_indexes],
        [kept_ids[i] for i in test_indexes],
        cache_name="AudioMNIST", cache_split="test", **feature_kwargs)

    _standardize_audio_features(training_dataset, testing_dataset)

    return training_dataset, testing_dataset


def get_RAVDESS_dataset(audio_dir=r'./dataset/RAVDESS',
                        test_ratio=0.2, seed=42,
                        target_frames=AUDIO_RAVDESS_FRAMES,
                        sample_rate=AUDIO_SAMPLE_RATE, n_mels=AUDIO_N_MELS,
                        n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH,
                        top_db=AUDIO_TOP_DB):
    """RAVDESS 情感语音数据集（AUD_CLF）。

    - 数据：``Audio_Speech_Actors_01-24``（1440 条语音、24 位演员，48 kHz WAV），
      仅使用 audio-only + speech（modality=03、vocal=01），歌曲与视频文件忽略；
    - 标签：与 CREMA-D 同口径的 HAP-vs-rest 二分类，``happy``（emotion=03）= 1，
      其余 7 类（neutral/calm/sad/angry/fearful/disgust/surprised）= 0；
    - 敏感属性：性别，演员编号为奇数 → ``Male`` = 1，偶数 → ``Female`` = 0；
    - 划分：按演员（说话人）划分，并按性别分层，训练/测试说话人互斥。

    文件命名规则为
    ``{modality}-{vocal}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav``。

    为什么选 RAVDESS / Why RAVDESS
    ------------------------------
    CREMA-D 只有 91 位演员、且种族分组为 61 : 29；RAVDESS 提供 24 位演员（12 男/12 女）
    的公开语音情感数据，任务定义与 CREMA-D 完全同构（HAP-vs-rest + 性别敏感属性），
    可作为音频侧的第三个独立数据集，用于检验结论是否只在 CREMA-D 上成立。
    其 24 位演员规模适合按说话人划分客户端（K=20，每轮按 FL_fraction 采样）。
    """
    relative_paths = []
    for root, _dirs, files in os.walk(audio_dir):
        for file_name in files:
            if file_name.lower().endswith('.wav'):
                relative_paths.append(
                    os.path.relpath(os.path.join(root, file_name), audio_dir)
                )
    relative_paths.sort()

    labels, protected, speaker_ids = [], [], []
    speaker_groups = {}
    kept_names, kept_ids = [], []
    dropped = 0
    for relative_path in relative_paths:
        parts = os.path.splitext(os.path.basename(relative_path))[0].split('-')
        if len(parts) != 7:
            dropped += 1
            continue
        modality, vocal, emotion, _intensity, _statement, _repetition, actor = parts
        # 只保留 audio-only（03）的语音（01），跳过歌曲（02）与非音频模态
        if modality != '03' or vocal != '01':
            dropped += 1
            continue
        try:
            actor_id = str(int(actor)).zfill(2)
        except ValueError:
            dropped += 1
            continue

        is_male = int(actor_id) % 2 == 1
        kept_names.append(relative_path)
        kept_ids.append(relative_path)
        labels.append(1 if emotion == '03' else 0)
        protected.append(1 if is_male else 0)
        speaker_ids.append(actor_id)
        speaker_groups[actor_id] = 'Male' if is_male else 'Female'

    if dropped:
        print(f"[RAVDESS] Dropped {dropped} files (non-speech song/video or unparsable name)")

    train_speakers, test_speakers = _stratified_speaker_split(
        speaker_groups, test_ratio, seed
    )
    print(f"[RAVDESS] Speaker-level split: {len(train_speakers)} train / "
          f"{len(test_speakers)} test speakers; "
          f"{len(labels)} speech clips kept")

    feature_kwargs = dict(target_frames=target_frames, sample_rate=sample_rate,
                          n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                          top_db=top_db)
    train_set, test_set = set(train_speakers), set(test_speakers)
    train_indexes = [i for i, speaker in enumerate(speaker_ids) if speaker in train_set]
    test_indexes = [i for i, speaker in enumerate(speaker_ids) if speaker in test_set]

    training_dataset = CustomizedAudioDataset(
        audio_dir, [kept_names[i] for i in train_indexes],
        [labels[i] for i in train_indexes], [protected[i] for i in train_indexes],
        [kept_ids[i] for i in train_indexes],
        cache_name="RAVDESS", cache_split="train",
        speaker_ids=[speaker_ids[i] for i in train_indexes], **feature_kwargs)
    testing_dataset = CustomizedAudioDataset(
        audio_dir, [kept_names[i] for i in test_indexes],
        [labels[i] for i in test_indexes], [protected[i] for i in test_indexes],
        [kept_ids[i] for i in test_indexes],
        cache_name="RAVDESS", cache_split="test",
        speaker_ids=[speaker_ids[i] for i in test_indexes], **feature_kwargs)

    _standardize_audio_features(training_dataset, testing_dataset)

    return training_dataset, testing_dataset


# Some codes are borrow from https://github.com/optimization-for-data-driven-science/Renyi-Fair-Inference
def get_ADULT_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag, use_csv_file=True, do_pca=False,
                      pca_dimension=32):
    # Using the csv files provided by Renyi-Fair-Inference (https://github.com/optimization-for-data-driven-science/Renyi-Fair-Inference)
    if use_csv_file:
        # Loading the label and sensitive attribute in training set
        with open(os.path.join(data_path, 'adult.data')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            y = []
            s1 = []
            s2 = []

            i = 0
            for row in csv_reader:
                if i == 0:
                    i += 1
                    continue

                if (row[9] == "Male") or ("Male" in row[9]):
                    s1.append(1)
                else:
                    s1.append(0)

                if (row[8] == "White") or ("White" in row[8]):
                    s2.append(1)
                else:
                    s2.append(0)

                if (row[14] == '>50K') or ('>50K' in row[14]):
                    y.append(1)
                else:
                    y.append(0)

        # Loading the label and sensitive attribute in test set
        with open(os.path.join(data_path, 'adult.test')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            testY = []
            testS1 = []
            testS2 = []
            i = 0
            for row in csv_reader:
                if i == 0:
                    i += 1
                    continue

                if (row[9] == "Male") or ("Male" in row[9]):
                    testS1.append(1)
                else:
                    testS1.append(0)

                if (row[8] == "White") or ("White" in row[8]):
                    testS2.append(1)
                else:
                    testS2.append(0)

                if (row[14] == '>50K') or ('>50K' in row[14]):
                    testY.append(1)
                else:
                    testY.append(0)

        with open(os.path.join(data_path, 'AdultTrain.csv')) as csv_file:
            csv_reader = csv.reader(csv_file)
            X = []
            i = 0
            for row in csv_reader:
                if i == 0:
                    i += 1
                    continue

                new_row = []
                for item in row:
                    new_row.append(float(item))

                new_row.append(1)  # intercept
                X.append(new_row)

        with open(os.path.join(data_path, 'AdultTest.csv')) as csv_file:
            csv_reader = csv.reader(csv_file)

            testX = []
            i = 0
            for row in csv_reader:
                if i == 0:
                    i += 1
                    continue

                new_row = []
                for item in row:
                    new_row.append(float(item))

                new_row.append(1)  # intercept

                testX.append(new_row)

        X = normalize(X, axis=0)
        testX = normalize(testX, axis=0)

        # Constructing the training dataset
        training_attribute_dict = {
            'X': X, 's1': s1, 's2': s2, 'y': y
        }
        training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

        # Constructing the positive and negative training dataset
        positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []

        # positive data point index in s1
        positive_array = (np.array(s1) == 1)
        for index, item in enumerate(positive_array):
            if item:
                positive_X.append(training_attribute_dict["X"][index])
                positive_s2.append(training_attribute_dict["s2"][index])
                positive_y.append(training_attribute_dict["y"][index])
            else:
                negative_X.append(training_attribute_dict["X"][index])
                negative_s2.append(training_attribute_dict["s2"][index])
                negative_y.append(training_attribute_dict["y"][index])

        positive_training_attribute_dict = {
            "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
        }
        negative_training_attribute_dict = {
            "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
        }
        positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
        negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

        # Constructing the testing dataset
        testing_attribute_dict = {
            'X': testX, 's1': testS1, 's2': testS2, 'y': testY
        }
        testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    # Preprocessing from raw data
    else:
        enc = OneHotEncoder()
        # Added the function of dimensionality reduction using PCA
        if do_pca:
            pca = PCA(n_components=pca_dimension)

        # Preprocess (training dataset)
        raw_X, raw_X_mask_s1, raw_X_mask_s2, raw_X_mask_s1_s2 = [], [], [], []
        y = []  # (Training set)Income over 50K (T:1, F:0)
        s1 = []  # (Training set)Sensitive feature (Male:1, Femal:0)
        s2 = []  # (Training set)Sensitive feature (White:1, non-White:0)

        with open(os.path.join(data_path, 'adult.data')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0:  # Skipping the row of feature name
                    continue
                if (row[9] == "Male") or ("Male" in row[9]):
                    s1.append(1)
                else:
                    s1.append(0)

                if (row[8] == "White") or ("White" in row[8]):
                    s2.append(1)
                else:
                    s2.append(0)

                if '>50K' in row[14]:
                    y.append(1)
                else:
                    y.append(0)

                row_copy = row[:14]
                row_mask_s1_copy = row[:9] + row[10:14]
                row_mask_s2_copy = row[:8] + row[9:14]
                row_mask_s1_s2_copy = row[:8] + row[10:14]

                raw_X.append(row_copy)
                raw_X_mask_s1.append(row_mask_s1_copy)
                raw_X_mask_s2.append(row_mask_s2_copy)
                raw_X_mask_s1_s2.append(row_mask_s1_s2_copy)

        # Preprocess (testing dataset)
        raw_testX, raw_testX_mask_s1, raw_testX_mask_s2, raw_testX_mask_s1_s2 = [], [], [], []
        testY = []  # (Testing set)Income over 50K (T:1, F:0)
        testS1 = []  # (Testing set)Sensitive feature (Male:1, Female:0)
        testS2 = []  # (Testing set)Sensitive feature (White:1, non-White:0)

        with open(os.path.join(data_path, 'adult.test')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0:  # Skipping the row of feature name
                    continue

                if (row[9] == "Male") or ("Male" in row[9]):
                    testS1.append(1)
                else:
                    testS1.append(0)

                if (row[8] == "White") or ("White" in row[8]):
                    testS2.append(1)
                else:
                    testS2.append(0)

                if '>50K' in row[14]:
                    testY.append(1)
                else:
                    testY.append(0)

                row_copy = row[:14]
                row_mask_s1_copy = row[:9] + row[10:14]
                row_mask_s2_copy = row[:8] + row[9:14]
                row_mask_s1_s2_copy = row[:8] + row[10:14]

                raw_testX.append(row_copy)
                raw_testX_mask_s1.append(row_mask_s1_copy)
                raw_testX_mask_s2.append(row_mask_s2_copy)
                raw_testX_mask_s1_s2.append(row_mask_s1_s2_copy)

        for data_index in range(len(raw_X)):
            for inner_index in range(len(raw_X[data_index])):
                if inner_index in [0, 2, 4, 10, 11, 12]:
                    raw_X[data_index][inner_index] = float(raw_X[data_index][inner_index])

        for data_index in range(len(raw_testX)):
            for inner_index in range(len(raw_X[data_index])):
                if inner_index in [0, 2, 4, 10, 11, 12]:
                    raw_testX[data_index][inner_index] = float(raw_testX[data_index][inner_index])

        # One-hot Encoding & PCA dimensionality reduction (training_dataset)
        enc.fit(raw_X + raw_testX)
        if mask_s1_flag:
            X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
            X_mask_s1 = np.float32(np.append(X_mask_s1_s2, np.array([s2]).transpose(), axis=1))
        elif mask_s2_flag:
            X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
            X_mask_s2 = np.float32(np.append(X_mask_s1_s2, np.array([s1]).transpose(), axis=1))
        elif mask_s1_s2_flag:
            X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        else:
            X = np.float32(enc.transform(raw_X).toarray())

        # One-hot Encoding (testing)
        if mask_s1_flag:
            testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
            testX_mask_s1 = np.float32(np.append(testX_mask_s1_s2, np.array([testS2]).transpose(), axis=1))
        elif mask_s2_flag:
            testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
            testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
        elif mask_s1_s2_flag:
            testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        else:
            testX = np.float32(enc.transform(raw_testX).toarray())
            # testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))

        # Constructing the training dataset
        training_attribute_dict = {
            # 'raw_X': np.array(raw_X), 'raw_X_mask_s1': np.array(raw_X_mask_s1),
            # 'raw_X_mask_s2': np.array(raw_X_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_X_mask_s1_s2),
            's1': s1, 's2': s2, 'y': y
        }
        if mask_s1_flag:
            if do_pca:
                pca.fit(X_mask_s1)
                training_attribute_dict['X'] = pca.transform(X_mask_s1)
            else:
                training_attribute_dict['X'] = X_mask_s1
        elif mask_s2_flag:
            if do_pca:
                pca.fit(X_mask_s2)
                training_attribute_dict['X'] = pca.transform(X_mask_s2)
            else:
                training_attribute_dict['X'] = X_mask_s2
        elif mask_s1_s2_flag:
            if do_pca:
                pca.fit(X_mask_s1_s2)
                training_attribute_dict['X'] = pca.transform(X_mask_s1_s2)
            else:
                training_attribute_dict['X'] = X_mask_s1_s2
        else:
            if do_pca:
                pca.fit(X)
                training_attribute_dict['X'] = pca.transform(X)
            else:
                training_attribute_dict['X'] = X

        training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

        # Constructing the positive and negative training dataset
        positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []
        # positive data point index of ndarry s1
        positive_array = (np.array(s1) == 1)
        for index, item in enumerate(positive_array):
            if item:
                positive_X.append(training_attribute_dict["X"][index])
                positive_s2.append(training_attribute_dict["s2"][index])
                positive_y.append(training_attribute_dict["y"][index])
            else:
                negative_X.append(training_attribute_dict["X"][index])
                negative_s2.append(training_attribute_dict["s2"][index])
                negative_y.append(training_attribute_dict["y"][index])

        positive_training_attribute_dict = {
            "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
        }
        negative_training_attribute_dict = {
            "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
        }
        positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
        negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

        # Constructing the testing dataset
        testing_attribute_dict = {
            # 'raw_X': np.array(raw_testX), 'raw_X_mask_s1': np.array(raw_testX_mask_s1),
            # 'raw_X_mask_s2': np.array(raw_testX_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_testX_mask_s1_s2),
            's1': testS1, 's2': testS2, 'y': testY
        }
        if mask_s1_flag:
            if do_pca:
                pca.fit(testX_mask_s1)
                testing_attribute_dict['X'] = pca.transform(testX_mask_s1)
            else:
                testing_attribute_dict['X'] = testX_mask_s1
        elif mask_s2_flag:
            if do_pca:
                pca.fit(testX_mask_s2)
                testing_attribute_dict['X'] = pca.transform(testX_mask_s2)
            else:
                testing_attribute_dict['X'] = testX_mask_s2
        elif mask_s1_s2_flag:
            if do_pca:
                pca.fit(testX_mask_s1_s2)
                testing_attribute_dict['X'] = pca.transform(testX_mask_s1_s2)
            else:
                testing_attribute_dict['X'] = testX_mask_s1_s2
        else:
            if do_pca:
                pca.fit(testX)
                testing_attribute_dict['X'] = pca.transform(testX)
            else:
                testing_attribute_dict['X'] = testX

        testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset


def get_ARRHYTHMIA_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag):
    # Preprocess
    full_X = []
    full_y = []  # Distinguish between the presence and absence of cardiac arrhythmia ('1': 1, '2'-'16':0)
    full_s = []  # Sensitive feature (Male:0, Female:1)

    with open(os.path.join(data_path, 'arrhythmia.data')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            temp = row[:13] + row[14:-1]
            try:
                full_X.append([float(item) for item in temp])
            except Exception:
                continue

            if ("1" in row[1]) or (int(row[1]) - 1 == 0):
                full_s.append(float(1))
            else:
                full_s.append(float(0))

            if int(row[-1]) == 1:
                full_y.append(float(1))
            else:
                full_y.append(float(0))

    training_size = int(len(full_X) * 0.8)
    training_indexes = random.sample(range(0, len(full_X)), training_size)
    X, y, s = [], [], []
    testX, testY, testS = [], [], []
    for i, item in enumerate(full_X):
        if i in training_indexes:
            X.append(item)
            y.append(full_y[i])
            s.append(full_s[i])
        else:
            testX.append(item)
            testY.append(full_y[i])
            testS.append(full_s[i])
    X, testX = np.array(X), np.array(testX)
    # Constructing the training dataset
    training_attribute_dict = {'X': X, 's1': s, 's2': s, 'y': y}
    training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

    # Constructing the positive and negative training dataset
    positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []
    # positive data point index of ndarry s1
    positive_array = (np.array(s) == 1)
    for index, item in enumerate(positive_array):
        if item:
            positive_X.append(training_attribute_dict["X"][index])
            positive_s2.append(training_attribute_dict["s2"][index])
            positive_y.append(training_attribute_dict["y"][index])
        else:
            negative_X.append(training_attribute_dict["X"][index])
            negative_s2.append(training_attribute_dict["s2"][index])
            negative_y.append(training_attribute_dict["y"][index])

    positive_training_attribute_dict = {
        "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
    }
    negative_training_attribute_dict = {
        "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
    }
    positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
    negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {'X': testX, 's1': testS, 's2': testS, 'y': testY}
    testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset


def get_BANK_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag):
    # Some codes are borrow from https://github.com/optimization-for-data-driven-science/Renyi-Fair-Inference

    # Preprocess
    full_X = []
    full_y = []  # Client will subscribe a term deposit (Yes: 1, No:0)
    full_s = []  # Sensitive feature (Married:1, Other:0)

    with open(os.path.join(data_path, 'bank-full.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        i = 0
        for row in csv_reader:
            if i == 0:
                i += 1
                continue

            if (row[2] == "married") or ("married" in row[2]):
                full_s.append(1)
            else:
                full_s.append(0)

            if (row[16] == 'yes') or ('yes' in row[16]):
                full_y.append(float(1))
            else:
                full_y.append(float(0))

    with open(os.path.join(data_path, 'Bank_data.csv')) as csv_file:
        csv_reader = csv.reader(csv_file)
        for _, row in enumerate(csv_reader):
            if _ == 0:
                continue
            new_row = []
            for item in row:
                new_row.append(float(item))
            full_X.append(new_row)
    # Copy from the description of R´E NYI FAIR INFERENCE
    training_size = 32000
    training_indexes = random.sample(range(0, len(full_X)), training_size)
    X, y, s = [], [], []
    testX, testY, testS = [], [], []
    for i, item in enumerate(full_X):
        if i in training_indexes:
            X.append(item)
            y.append(full_y[i])
            s.append(full_s[i])
        else:
            testX.append(item)
            testY.append(full_y[i])
            testS.append(full_s[i])
    X, testX = np.array(X), np.array(testX)
    # BANK feature standardisation (z-score), fitted on the training split only.
    # Bank_data.csv mixes one-hot 0/1 columns with raw-scale numerics (balance up to
    # ~1e5, std ~3e3), so an unscaled MLP saturates: the training loss freezes at a
    # constant value across every client/epoch and the model collapses to the
    # majority class (DEO 0.0, FR 1.0, SPD 0.0). Standardising matches the scaled
    # views the other tabular loaders use (COMPAS/DRUG via PCA, ADULT via normalize).
    # BANK 特征标准化（z-score），仅在训练集上拟合；Bank_data.csv 混合了 one-hot 0/1 列
    # 与原始量纲数值列（balance 最大 ~1e5、标准差 ~3e3），未缩放时 MLP 会饱和：训练损失在
    # 各客户端/各 epoch 上冻结为同一常数，模型坍缩到多数类（DEO 0.0、FR 1.0、SPD 0.0）；
    # 标准化后与其余表格加载器的尺度处理保持一致（COMPAS/DRUG 经 PCA、ADULT 经 normalize）。
    _bank_mean = X.mean(axis=0, keepdims=True)
    _bank_std = X.std(axis=0, keepdims=True)
    _bank_std[_bank_std == 0.0] = 1.0
    X = np.float32((X - _bank_mean) / _bank_std)
    testX = np.float32((testX - _bank_mean) / _bank_std)
    # Constructing the training dataset
    training_attribute_dict = {'X': X, 's1': s, 's2': s, 'y': y}
    training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

    # Constructing the positive and negative training dataset
    positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []
    # positive data point index of ndarry s1
    positive_array = (np.array(s) == 1)
    for index, item in enumerate(positive_array):
        if item:
            positive_X.append(training_attribute_dict["X"][index])
            positive_s2.append(training_attribute_dict["s2"][index])
            positive_y.append(training_attribute_dict["y"][index])
        else:
            negative_X.append(training_attribute_dict["X"][index])
            negative_s2.append(training_attribute_dict["s2"][index])
            negative_y.append(training_attribute_dict["y"][index])

    positive_training_attribute_dict = {
        "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
    }
    negative_training_attribute_dict = {
        "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
    }
    positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
    negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {'X': testX, 's1': testS, 's2': testS, 'y': testY}
    testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset


def get_COMPAS_dataset(data_path, mask_s1_flag=False, mask_s2_flag=False, mask_s1_s2_flag=False):
    # Some codes are borrow from https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    enc = OneHotEncoder()
    pca = PCA(n_components=64)

    with open(os.path.join(data_path, 'compas-scores-two-years.csv')) as csv_file:

        csv_reader = csv.reader(csv_file)
        raw_data = []

        # Filtering
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skipping the row of feature name
                continue

            if row[15] != '' and row[24] != '' and row[22] != '' and row[40] != '':
                if 30 >= int(row[15]) >= -30:  # Filtering by `days_b_screening_arrest`
                    if int(row[24]) != -1:  # Filtering by `is_recid`
                        if row[22] != "0":  # Filtering by `c_charge_degree`
                            if row[40] != 'N/A':  # Filtering by `score_text`
                                if row[9] == "African-American" or row[9] == "Caucasian":  # Filtering by `race`
                                    raw_data.append(row)

        # Splitting
        random.seed(42)
        random.shuffle(raw_data)
        training_set = raw_data[:4800]
        testing_set = raw_data[4800:]

        # Training set
        raw_X, raw_X_mask_s1, raw_X_mask_s2, raw_X_mask_s1_s2 = [], [], [], []
        y = []  # (Training set)Not a recidivist (is_recid=0 -> 1; is_recid=1 -> 0)
        s1 = []  # (Training set)Sensitive feature (African-American:1, Caucasian:0)
        s2 = []
        for i, row in enumerate(training_set):
            if row[9] == "African-American":  # African-American:1, Caucasian:0
                s1.append(1)
            else:
                s1.append(0)

            if row[5] == "Male":  # Male:1, Female:0
                s2.append(1)
            else:
                s2.append(0)

            if int(row[24]) == 0:  # Not a recidivist (is_recid=0 -> 1; is_recid=1 -> 0)
                y.append(1)
            else:
                y.append(0)

            row_copy = row[5:6] + row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[
                                                                                                             48:49]  # Filtering out excess features
            row_mask_s1_copy = row[5:6] + row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s2_copy = row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s1_s2_copy = row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]

            # row_copy = row[:24] + row[25:-1]  # Filtering the label and the feature 'two_year_recid' in last column
            # row_mask_s1_copy = row[:9] + row[10:24] + row[25:-1]
            # row_mask_s2_copy = row[:5] + row[6:24] + row[25:-1]
            # row_mask_s1_s2_copy = row[:5] + row[6:8] + row[10:24] + row[25:-1]

            raw_X.append(row_copy)
            raw_X_mask_s1.append(row_mask_s1_copy)
            raw_X_mask_s2.append(row_mask_s2_copy)
            raw_X_mask_s1_s2.append(row_mask_s1_s2_copy)

        # Testing
        raw_testX, raw_testX_mask_s1, raw_testX_mask_s2, raw_testX_mask_s1_s2 = [], [], [], []
        testY = []  # (Testing set)Not a recidivist (T:1, F:0)
        testS1 = []  # (Testing set)Sensitive feature (African-American:1, Caucasian:0)
        testS2 = []  # (Testing set)Sensitive feature (Male:1, Female:0)

        for i, row in enumerate(testing_set):
            if row[9] == "African-American":  # African-American:1, Caucasian:0
                testS1.append(1)
            else:
                testS1.append(0)

            if row[5] == "Male":  # Male:1, Female:0
                testS2.append(1)
            else:
                testS2.append(0)

            if int(row[24]) == 0:  # Not a recidivist (is_recid=0->T:1; is_recid=1->F:0)
                testY.append(1)
            else:
                testY.append(0)

            row_copy = row[5:6] + row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[
                                                                                                             48:49]  # Filtering out excess features
            row_mask_s1_copy = row[5:6] + row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s2_copy = row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s1_s2_copy = row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]

            # row_copy = row[:24] + row[25:-1]  # Filtering the label and the feature 'two_year_recid' in last column
            # row_mask_s1_copy = row[:9] + row[10:24] + row[25:-1]
            # row_mask_s2_copy = row[:5] + row[6:24] + row[25:-1]
            # row_mask_s1_s2_copy = row[:5] + row[6:8] + row[10:24] + row[25:-1]

            raw_testX.append(row_copy)
            raw_testX_mask_s1.append(row_mask_s1_copy)
            raw_testX_mask_s2.append(row_mask_s2_copy)
            raw_testX_mask_s1_s2.append(row_mask_s1_s2_copy)

    # One-hot Encoding (training_dataset)
    enc.fit(raw_X_mask_s1_s2 + raw_testX_mask_s1_s2)

    if mask_s1_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s1 = np.float32(np.append(X_mask_s1_s2, np.array([s2]).transpose(), axis=1))
    elif mask_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s2 = np.float32(np.append(X_mask_s1_s2, np.array([s1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
    else:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X = np.float32(np.append(X_mask_s1_s2, np.array([s1, s2]).transpose(), axis=1))

    # One-hot Encoding (testing)
    if mask_s1_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s1 = np.float32(np.append(testX_mask_s1_s2, np.array([testS2]).transpose(), axis=1))
    elif mask_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
    else:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        # testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
        testX = np.float32(np.append(testX_mask_s1_s2, np.array([testS1, testS2]).transpose(), axis=1))

    # Constructing the training dataset
    training_attribute_dict = {
        # 'raw_X': np.array(raw_X), 'raw_X_mask_s1': np.array(raw_X_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_X_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_X_mask_s1_s2),
        's1': s1, 's2': s2, 'y': y
    }
    if mask_s1_flag:
        pca.fit(X_mask_s1)
        training_attribute_dict['X'] = pca.transform(X_mask_s1)
    elif mask_s2_flag:
        pca.fit(X_mask_s2)
        training_attribute_dict['X'] = pca.transform(X_mask_s2)
    elif mask_s1_s2_flag:
        pca.fit(X_mask_s1_s2)
        training_attribute_dict['X'] = pca.transform(X_mask_s1_s2)
    else:
        pca.fit(X)
        training_attribute_dict['X'] = pca.transform(X)

    training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

    # Constructing the positive and negative training dataset
    positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []
    # positive data point index of ndarry s1
    positive_array = (np.array(s1) == 1)
    for index, item in enumerate(positive_array):
        if item:
            positive_X.append(training_attribute_dict["X"][index])
            positive_s2.append(training_attribute_dict["s2"][index])
            positive_y.append(training_attribute_dict["y"][index])
        else:
            negative_X.append(training_attribute_dict["X"][index])
            negative_s2.append(training_attribute_dict["s2"][index])
            negative_y.append(training_attribute_dict["y"][index])

    positive_training_attribute_dict = {
        "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
    }
    negative_training_attribute_dict = {
        "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
    }
    positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
    negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {
        # 'raw_X': np.array(raw_testX), 'raw_X_mask_s1': np.array(raw_testX_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_testX_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_testX_mask_s1_s2),
        's1': testS1, 's2': testS2, 'y': testY
    }
    if mask_s1_flag:
        pca.fit(testX_mask_s1)
        testing_attribute_dict['X'] = pca.transform(testX_mask_s1)
    elif mask_s2_flag:
        pca.fit(testX_mask_s2)
        testing_attribute_dict['X'] = pca.transform(testX_mask_s2)
    elif mask_s1_s2_flag:
        pca.fit(testX_mask_s1_s2)
        testing_attribute_dict['X'] = pca.transform(testX_mask_s1_s2)
    else:
        pca.fit(testX)
        testing_attribute_dict['X'] = pca.transform(testX)

    testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset


def get_DRUG_dataset(data_path, mask_s1_flag=False, mask_s2_flag=False, mask_s1_s2_flag=False):
    enc = OneHotEncoder()
    pca = PCA(n_components=64)

    with open(os.path.join(data_path, 'drug_consumption.data')) as csv_file:
        csv_reader = csv.reader(csv_file)
        raw_data = []

        # Pre_process: Filtering
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skipping the row of feature name
                continue
            raw_data.append(row)

        # Splitting
        random.seed(42)
        random.shuffle(raw_data)
        training_set = raw_data[:1600]
        testing_set = raw_data[1600:]

        # Training set
        raw_X, raw_X_mask_s1, raw_X_mask_s2, raw_X_mask_s1_s2 = [], [], [], []
        y = []  # (Training set)Not abuse volatile substance (Not abuse:1 ; Abuse:0)
        s1 = []  # (Training set)Sensitive feature (White:1, Non-white:0)
        s2 = []  # (Training set)Sensitive feature (Male:1, Female:0)
        for i, row in enumerate(training_set):
            if float(row[5]) == -0.31685:  # White:1, Non-white:0
                s1.append(1)
            else:
                s1.append(0)

            if float(row[2]) < 0:  # Male:1, Female:0
                s2.append(1)
            else:
                s2.append(0)

            if row[31] == 'CL0':  # Not abuse volatile substance (Not abuse:1 ; Abuse:0)
                y.append(1)
            else:
                y.append(0)

            row_copy = row[:31]  # Filtering the label in last column
            row_mask_s1_copy = row[:5] + row[6:31]
            row_mask_s2_copy = row[:2] + row[3:31]
            row_mask_s1_s2_copy = row[:2] + row[3:5] + row[6:31]

            raw_X.append(row_copy)
            raw_X_mask_s1.append(row_mask_s1_copy)
            raw_X_mask_s2.append(row_mask_s2_copy)
            raw_X_mask_s1_s2.append(row_mask_s1_s2_copy)

        # Testing
        raw_testX, raw_testX_mask_s1, raw_testX_mask_s2, raw_testX_mask_s1_s2 = [], [], [], []
        testY = []  # (Testing set)Not abuse volatile substance (Not abuse:1 ; Abuse:0)
        testS1 = []  # (Testing set)Sensitive feature (White:1, Non-white:0)
        testS2 = []  # (Testing set)Sensitive feature (Male:1, Female:0)

        for i, row in enumerate(testing_set):
            if float(row[5]) == -0.31685:  # White:1, Non-white:0
                testS1.append(1)
            else:
                testS1.append(0)

            if float(row[2]) < 0:  # Male:1, Female:0
                testS2.append(1)
            else:
                testS2.append(0)

            if row[31] == 'CL0':  # Not abuse volatile substance (Not abuse:1 ; Abuse:0)
                testY.append(1)
            else:
                testY.append(0)

            row_copy = row[:31]  # Filtering the label in last column
            row_mask_s1_copy = row[:5] + row[6:31]
            row_mask_s2_copy = row[:2] + row[3:31]
            row_mask_s1_s2_copy = row[:2] + row[3:5] + row[6:31]

            raw_testX.append(row_copy)
            raw_testX_mask_s1.append(row_mask_s1_copy)
            raw_testX_mask_s2.append(row_mask_s2_copy)
            raw_testX_mask_s1_s2.append(row_mask_s1_s2_copy)

    # One-hot Encoding (training_dataset)
    enc.fit(raw_X_mask_s1_s2 + raw_testX_mask_s1_s2)
    if mask_s1_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s1 = np.float32(np.append(X_mask_s1_s2, np.array([s2]).transpose(), axis=1))
    elif mask_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s2 = np.float32(np.append(X_mask_s1_s2, np.array([s1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
    else:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X = np.float32(np.append(X_mask_s1_s2, np.array([s1, s2]).transpose(), axis=1))

    # One-hot Encoding (testing)
    if mask_s1_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s1 = np.float32(np.append(testX_mask_s1_s2, np.array([testS2]).transpose(), axis=1))
    elif mask_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
    else:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        # testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
        testX = np.float32(np.append(testX_mask_s1_s2, np.array([testS1, testS2]).transpose(), axis=1))

    # Constructing the training dataset
    training_attribute_dict = {
        # 'raw_X': np.array(raw_X), 'raw_X_mask_s1': np.array(raw_X_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_X_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_X_mask_s1_s2),
        's1': s1, 's2': s2, 'y': y
    }
    if mask_s1_flag:
        pca.fit(X_mask_s1)
        training_attribute_dict['X'] = pca.transform(X_mask_s1)
    elif mask_s2_flag:
        pca.fit(X_mask_s2)
        training_attribute_dict['X'] = pca.transform(X_mask_s2)
    elif mask_s1_s2_flag:
        pca.fit(X_mask_s1_s2)
        training_attribute_dict['X'] = pca.transform(X_mask_s1_s2)
    else:
        pca.fit(X)
        training_attribute_dict['X'] = pca.transform(X)

    training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

    # Constructing the positive and negative training dataset
    positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []
    # positive data point index of ndarry s1
    positive_array = (np.array(s1) == 1)
    for index, item in enumerate(positive_array):
        if item:
            positive_X.append(training_attribute_dict["X"][index])
            positive_s2.append(training_attribute_dict["s2"][index])
            positive_y.append(training_attribute_dict["y"][index])
        else:
            negative_X.append(training_attribute_dict["X"][index])
            negative_s2.append(training_attribute_dict["s2"][index])
            negative_y.append(training_attribute_dict["y"][index])

    positive_training_attribute_dict = {
        "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
    }
    negative_training_attribute_dict = {
        "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
    }
    positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
    negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {
        # 'raw_X': np.array(raw_testX), 'raw_X_mask_s1': np.array(raw_testX_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_testX_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_testX_mask_s1_s2),
        's1': testS1, 's2': testS2, 'y': testY
    }
    if mask_s1_flag:
        pca.fit(testX_mask_s1)
        testing_attribute_dict['X'] = pca.transform(testX_mask_s1)
    elif mask_s2_flag:
        pca.fit(testX_mask_s2)
        testing_attribute_dict['X'] = pca.transform(testX_mask_s2)
    elif mask_s1_s2_flag:
        pca.fit(testX_mask_s1_s2)
        testing_attribute_dict['X'] = pca.transform(testX_mask_s1_s2)
    else:
        pca.fit(testX)
        testing_attribute_dict['X'] = pca.transform(testX)

    testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset


def get_DUTCH_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag):
    # Preprocess
    enc = OneHotEncoder()

    full_X = []
    raw_X = []
    full_y = []  # Using occupation as the class label ('5_4_9': 1, '2_1':0)
    full_s = []  # Sensitive feature (2->Male:0, 1->Female:1)

    with open(os.path.join(data_path, 'dutch_census_2001.arff'), encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    df = np.array(df).tolist()

    for row in df:
        if math.isnan(row[0]):
            continue
        temp = row[:11]
        try:
            raw_X.append([float(item) for item in temp])
        except Exception:
            continue

        if ("1" in str(row[0])) or (int(row[0]) - 1) == 0:
            full_s.append(float(1))
        else:
            full_s.append(float(0))

        if (row[-1] == "5_4_9") or ("5_4_9" in row[-1]):
            full_y.append(float(1))
        else:
            full_y.append(float(0))

    enc.fit(raw_X)
    full_X = enc.transform(raw_X).toarray()

    training_size = int(len(full_X) * 0.8)
    random.seed(42)
    training_indexes = random.sample(range(0, len(full_X)), training_size)
    X, y, s = [], [], []
    testX, testY, testS = [], [], []
    for i, item in enumerate(full_X):
        if i in training_indexes:
            X.append(item)
            y.append(full_y[i])
            s.append(full_s[i])
        else:
            testX.append(item)
            testY.append(full_y[i])
            testS.append(full_s[i])
    X, testX = np.array(X), np.array(testX)
    # Constructing the training dataset
    # pca.fit(X)
    training_attribute_dict = {'s1': s, 's2': s, 'y': y}
    # training_attribute_dict['X'] = pca.transform(X)
    training_attribute_dict['X'] = X
    training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

    # Constructing the positive and negative training dataset
    positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []
    # positive data point index of ndarry s1
    positive_array = (np.array(s) == 1)
    for index, item in enumerate(positive_array):
        if item:
            positive_X.append(training_attribute_dict["X"][index])
            positive_s2.append(training_attribute_dict["s2"][index])
            positive_y.append(training_attribute_dict["y"][index])
        else:
            negative_X.append(training_attribute_dict["X"][index])
            negative_s2.append(training_attribute_dict["s2"][index])
            negative_y.append(training_attribute_dict["y"][index])

    positive_training_attribute_dict = {
        "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
    }
    negative_training_attribute_dict = {
        "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
    }
    positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
    negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

    # Constructing the testing dataset
    # pca.fit(testX)
    testing_attribute_dict = {'s1': testS, 's2': testS, 'y': testY}
    # testing_attribute_dict['X'] = pca.transform(testX)
    testing_attribute_dict['X'] = testX
    testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset


def get_GERMAN_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag):
    # Preprocess
    full_X = []
    full_y = []  # Good or bad credit risks (Good:1, Bad:0)
    full_s1 = []  # Sensitive feature: Gender (Female:1, Male:0)
    full_s2 = []  # Sensitive feature: Marital-status (Married:1, Other:0)

    with open(os.path.join(data_path, 'german.data')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            if ("A92" in row[8]) or ("A95" in row[8]) or (row[8] == "A92") or (row[8] == "A95"):
                full_s1.append(float(1))  # Female
            else:
                full_s1.append(float(0))  # Male

            if ("A92" in row[8]) or ("A94" in row[8]) or (row[8] == "A92") or (row[8] == "A94"):
                full_s2.append(float(1))  # Married
            else:
                full_s2.append(float(0))  # Other

            if ('1' in row[-1]) or (row[-1] == '1') or (int(row[-1]) - 1 == 0):
                full_y.append(float(1))  # Good
            else:
                full_y.append(float(0))  # Bad

    with open(os.path.join(data_path, 'german.data-numeric')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for _, row in enumerate(csv_reader):
            new_row = []
            for item in row:
                if (len(item) == 0) or (len(item) - 1 == -1):
                    continue
                new_row.append(float(item))
            full_X.append(new_row)
    # Copy from the description of Renyi(R´E NYI FAIR INFERENCE)
    training_size = int(len(full_X) * 0.8)
    training_indexes = random.sample(range(0, len(full_X)), training_size)
    X, y, s1, s2 = [], [], [], []
    testX, testY, testS1, testS2 = [], [], [], []
    for i, item in enumerate(full_X):
        if i in training_indexes:
            X.append(item)
            y.append(full_y[i])
            s1.append(full_s1[i])
            s2.append(full_s2[i])
        else:
            testX.append(item)
            testY.append(full_y[i])
            testS1.append(full_s1[i])
            testS2.append(full_s2[i])
    X, testX = np.array(X), np.array(testX)
    # Constructing the training dataset
    training_attribute_dict = {'X': X, 's1': s1, 's2': s2, 'y': y}
    training_dataset = CustomizedTabularDataset(attribute_dict=training_attribute_dict)

    # Constructing the positive and negative training dataset
    positive_X, negative_X, positive_y, negative_y, positive_s2, negative_s2 = [], [], [], [], [], []
    # positive data point index of ndarry s1
    positive_array = (np.array(s1) == 1)
    for index, item in enumerate(positive_array):
        if item:
            positive_X.append(training_attribute_dict["X"][index])
            positive_s2.append(training_attribute_dict["s2"][index])
            positive_y.append(training_attribute_dict["y"][index])
        else:
            negative_X.append(training_attribute_dict["X"][index])
            negative_s2.append(training_attribute_dict["s2"][index])
            negative_y.append(training_attribute_dict["y"][index])

    positive_training_attribute_dict = {
        "X": np.array(positive_X), 's1': [1 for i in range(len(positive_X))], 's2': positive_s2, 'y': positive_y
    }
    negative_training_attribute_dict = {
        "X": np.array(negative_X), 's1': [0 for i in range(len(negative_X))], 's2': negative_s2, 'y': negative_y
    }
    positive_training_dataset = CustomizedTabularDataset(attribute_dict=positive_training_attribute_dict)
    negative_training_dataset = CustomizedTabularDataset(attribute_dict=negative_training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {'X': testX, 's1': testS1, 's2': testS2, 'y': testY}
    testing_dataset = CustomizedTabularDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, positive_training_dataset, negative_training_dataset, testing_dataset

