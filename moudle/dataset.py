import os
import torch
import csv
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


def _get_cache_dir(dataset_name, split):
    cache_path = os.path.join(CACHE_DIR, dataset_name, split)
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def _shard_exists(cache_dir, total_len):
    if not os.path.exists(cache_dir):
        return False
    meta_path = os.path.join(cache_dir, "meta.pt")
    if not os.path.exists(meta_path):
        return False
    meta = torch.load(meta_path, weights_only=False)
    num_shards = meta["num_shards"]
    for i in range(num_shards):
        if not os.path.exists(os.path.join(cache_dir, f"shard_{i}.pt")):
            return False
    return True


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


def _load_shards(cache_dir):
    meta = torch.load(os.path.join(cache_dir, "meta.pt"), weights_only=False)
    num_shards = meta["num_shards"]
    items_list = []
    for i in range(num_shards):
        shard = torch.load(os.path.join(cache_dir, f"shard_{i}.pt"), weights_only=False)
        items_list.extend(shard)
    return items_list, meta


class CachedImageDataset(Dataset):
    def __init__(self, cache_dir):
        self.items, self.meta = _load_shards(cache_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CachedTextDataset(Dataset):
    def __init__(self, cache_dir):
        self.items, self.meta = _load_shards(cache_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
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
                print(f"[Cache] Loading {cache_name} {cache_split} from cache: {cache_dir}")
                cached = CachedImageDataset(cache_dir)
                self._cached_items = cached.items
                self._cache_meta = cached.meta
                self._use_cache = True
            else:
                print(f"[Cache] No valid cache found for {cache_name} {cache_split}, will build cache on first access")
                self._cache_dir = cache_dir
                self._use_cache = False
                self._cache_built = False
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
        shard_items = []
        skipped = 0
        total_valid = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, img_tensor in enumerate(executor.map(_load_image_worker, args_list, chunksize=256)):
                if img_tensor is not None:
                    shard_items.append({
                        'img': img_tensor,
                        'labels': torch.tensor(self.labels[i], dtype=torch.float),
                        'protected': torch.tensor(self.protected[i], dtype=torch.long)
                    })
                    total_valid += 1
                else:
                    skipped += 1

                if len(shard_items) >= MAX_SHARD_SIZE:
                    torch.save(shard_items, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
                    shard_items = []
                    shard_idx += 1

                if (i + 1) % 5000 == 0:
                    print(f"[Cache]   Processed {i + 1}/{len(self.img_names)} (skipped {skipped})")

        if shard_items:
            torch.save(shard_items, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
            shard_idx += 1

        meta = {"total_len": total_valid, "num_shards": shard_idx, "shard_size": MAX_SHARD_SIZE}
        torch.save(meta, os.path.join(self._cache_dir, "meta.pt"))

        self._cache_meta = meta
        self._use_cache = True
        self._cache_built = True
        if skipped > 0:
            print(f"[Cache]   Total skipped {skipped} corrupted images")
        print(f"[Cache] Image cache built successfully ({total_valid} valid samples)")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        if self._use_cache and hasattr(self, '_cached_items'):
            return self._cached_items[index]
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
            return self._shard_cache[shard_idx][item_idx]
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


# 训练集1613790条记录
# 测试集448276条记录
class MoJiDataset(Dataset):
    def __init__(self, texts, labels, protected, tokenizer, max_len, cache_name=None, cache_split=None):
        self.texts = texts
        self.labels = labels
        self.protected = protected
        self.tokenizer = tokenizer
        self.max_len = max_len

        if cache_name is not None and cache_split is not None:
            cache_dir = _get_cache_dir(cache_name, cache_split)
            if _shard_exists(cache_dir, len(texts)):
                print(f"[Cache] Loading {cache_name} {cache_split} from cache: {cache_dir}")
                cached = CachedTextDataset(cache_dir)
                self._cached_items = cached.items
                self._use_cache = True
            else:
                print(f"[Cache] No valid cache found for {cache_name} {cache_split}, will build cache on first access")
                self._cache_dir = cache_dir
                self._use_cache = False
                self._cache_built = False
        else:
            self._use_cache = False
            self._cache_built = False

    def _build_cache(self):
        print(f"[Cache] Building text cache to: {self._cache_dir} ({len(self.texts)} samples)")
        items = []
        for i in range(len(self.texts)):
            text = str(self.texts[i])
            label = self.labels[i]
            protected_label = self.protected[i]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            items.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long),
                'protected': torch.tensor(protected_label, dtype=torch.long)
            })
            if (i + 1) % 50000 == 0:
                print(f"[Cache]   Tokenized {i + 1}/{len(self.texts)}")
        _save_shards(self._cache_dir, items)
        self._cached_items = items
        self._use_cache = True
        self._cache_built = True
        print(f"[Cache] Text cache built successfully")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self._use_cache:
            return self._cached_items[item]
        if not self._cache_built and hasattr(self, '_cache_dir'):
            self._build_cache()
            return self._cached_items[item]
        text = str(self.texts[item])
        label = self.labels[item]
        protected_label = self.protected[item]

        encoding = self.tokenizer.encode_plus(
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
            'labels': torch.tensor(label, dtype=torch.long),
            'protected': torch.tensor(protected_label, dtype=torch.long)
        }


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


# hard_text文本内容，profession分类（0-27号分类），gender是敏感属性男0女1
# 训练集257478条记录
# 测试集99069条记录
class BiosDataset(Dataset):
    def __init__(self, texts, labels, protected, tokenizer, max_len, cache_name=None, cache_split=None):
        self.texts = texts
        self.labels = labels
        self.protected = protected
        self.tokenizer = tokenizer
        self.max_len = max_len

        if cache_name is not None and cache_split is not None:
            cache_dir = _get_cache_dir(cache_name, cache_split)
            if _shard_exists(cache_dir, len(texts)):
                print(f"[Cache] Loading {cache_name} {cache_split} from cache: {cache_dir}")
                cached = CachedTextDataset(cache_dir)
                self._cached_items = cached.items
                self._use_cache = True
            else:
                print(f"[Cache] No valid cache found for {cache_name} {cache_split}, will build cache on first access")
                self._cache_dir = cache_dir
                self._use_cache = False
                self._cache_built = False
        else:
            self._use_cache = False
            self._cache_built = False

    def _build_cache(self):
        print(f"[Cache] Building text cache to: {self._cache_dir} ({len(self.texts)} samples)")
        items = []
        for i in range(len(self.texts)):
            text = str(self.texts[i])
            label = self.labels[i]
            protected_label = self.protected[i]

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            items.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long),
                'protected': torch.tensor(protected_label, dtype=torch.long)
            })
            if (i + 1) % 50000 == 0:
                print(f"[Cache]   Tokenized {i + 1}/{len(self.texts)}")
        _save_shards(self._cache_dir, items)
        self._cached_items = items
        self._use_cache = True
        self._cache_built = True
        print(f"[Cache] Text cache built successfully")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self._use_cache:
            return self._cached_items[item]
        if not self._cache_built and hasattr(self, '_cache_dir'):
            self._build_cache()
            return self._cached_items[item]
        text = str(self.texts[item])
        label = self.labels[item]
        protected_label = self.protected[item]

        encoding = self.tokenizer.encode_plus(
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
            'labels': torch.tensor(label, dtype=torch.long),
            'protected': torch.tensor(protected_label, dtype=torch.long)
        }

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
            print(f"[Cache] No valid cache found for CelebA {split}, will build cache on first access")
            self._cache_dir = cache_dir
            self._use_cache = False
            self._cache_built = False

    def _build_cache(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        print(f"[Cache] Building CelebA cache to: {self._cache_dir} ({len(self.img_names)} samples, {min(os.cpu_count() or 4, 4)} workers)")

        args_list = [
            (os.path.join(self.img_dir, self.img_names[i]), self.image_size)
            for i in range(len(self.img_names))
        ]

        num_workers = min(os.cpu_count() or 4, 4)
        shard_idx = 0
        shard_items = []
        skipped = 0
        total_valid = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, img_tensor in enumerate(executor.map(_load_image_worker, args_list, chunksize=256)):
                if img_tensor is not None:
                    shard_items.append({
                        'img': img_tensor,
                        'labels': torch.tensor(self.labels[i], dtype=torch.float),
                        'protected': torch.tensor(self.protected[i], dtype=torch.long)
                    })
                    total_valid += 1
                else:
                    skipped += 1

                if len(shard_items) >= MAX_SHARD_SIZE:
                    torch.save(shard_items, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
                    shard_items = []
                    shard_idx += 1

                if (i + 1) % 10000 == 0:
                    print(f"[Cache]   Processed {i + 1}/{len(self.img_names)} (skipped {skipped})")

        if shard_items:
            torch.save(shard_items, os.path.join(self._cache_dir, f"shard_{shard_idx}.pt"))
            shard_idx += 1

        meta = {"total_len": total_valid, "num_shards": shard_idx, "shard_size": MAX_SHARD_SIZE}
        torch.save(meta, os.path.join(self._cache_dir, "meta.pt"))

        self._cache_meta = meta
        self._use_cache = True
        self._cache_built = True
        if skipped > 0:
            print(f"[Cache]   Total skipped {skipped} corrupted images")
        print(f"[Cache] CelebA cache built successfully ({total_valid} valid samples)")

    def __getitem__(self, index):
        if self._use_cache and hasattr(self, '_cached_items'):
            return self._cached_items[index]
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
            return self._shard_cache[shard_idx][item_idx]
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

    training_img_names = os.listdir(img_dir+"fairface-img-margin025-trainval/train")
    training_labels, training_protected = [], []
    df = pd.read_csv(r'./dataset/FairFace/fairface_label_train.csv', encoding='utf-8')
    df_array = np.array(df)
    df_list = df_array.tolist()
    for item in df_list:
        gender = item[2]
        training_protected.append(1 if gender == 'Male' else 0)
        service_test = item[4]
        training_labels.append(1 if service_test else 0)

    test_img_names = os.listdir(img_dir + "fairface-img-margin025-trainval/val")
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
def get_LFWAPlus_dataset(img_dir=r'./dataset/LFWA+/', image_size=(250, 250)):
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

    # Constructing the training dataset
    training_dataset = CustomizedImageDataset('', training_img_names, training_labels, training_protected, transform, cache_name="LFWAPlus", cache_split="train")
    # Constructing the testing dataset
    testing_dataset = CustomizedImageDataset('', test_img_names, test_labels, test_protected, transform, cache_name="LFWAPlus", cache_split="test")

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

