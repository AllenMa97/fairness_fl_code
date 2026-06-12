# Moji
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from collections import defaultdict

moji_df_train = pd.read_parquet(r'dataset/moji/train.parquet')
moji_df_test = pd.read_parquet(r'dataset/moji/test.parquet')

bios_df_train = pd.read_parquet(r'dataset/bios/train.parquet')
bios_df_test = pd.read_parquet(r'dataset/bios/test.parquet')

# 编码标签
le = LabelEncoder()
moji_df_train['encoded_label'] = le.fit_transform(moji_df_train['label']) # moji
moji_df_test['encoded_label'] = le.transform(moji_df_test['label']) # moji

bios_df_train['encoded_label'] = le.fit_transform(bios_df_train['profession']) # bios
bios_df_test['encoded_label'] = le.transform(bios_df_test['profession']) # bios

moji_train_labels = moji_df_train['encoded_label'].tolist()
bios_train_labels = bios_df_train['encoded_label'].tolist()

moji_train_sa = moji_df_train['sa'].tolist() # moji
bios_train_sa = bios_df_train['gender'].tolist() # bios

moji_test_labels = moji_df_test['encoded_label'].tolist() # moji
bios_test_labels = bios_df_test['encoded_label'].tolist() # bios
moji_test_sa = moji_df_test['sa'].tolist() # moji
bios_test_sa = bios_df_test['gender'].tolist() # bios


# 示例数据
moji_train_data = []
moji_test_data = []
bios_train_data = []
bios_test_data = []

# moji
for index in range(0, len(moji_df_train)):
    tmp = {"group": moji_train_sa[index], "label":moji_train_labels[index]}
    moji_train_data.append(tmp)
for index in range(0, len(moji_df_test)):
    tmp = {"group": moji_test_sa[index], "label":moji_test_labels[index]}
    moji_test_data.append(tmp)

# bios
for index in range(0, len(bios_df_train)):
    tmp = {"group": bios_train_sa[index], "label":bios_train_labels[index]}
    bios_train_data.append(tmp)
for index in range(0, len(bios_df_test)):
    tmp = {"group": bios_test_sa[index], "label":bios_test_labels[index]}
    bios_test_data.append(tmp)


# 计算每个位置的重叠点数
moji_train_point_counts = defaultdict(int)
for point in moji_train_data:
    key = (point["group"], point["label"])
    moji_train_point_counts[key] += 1

moji_test_point_counts = defaultdict(int)
for point in moji_test_data:
    key = (point["group"], point["label"])
    moji_test_point_counts[key] += 1

bios_train_point_counts = defaultdict(int)
for point in bios_train_data:
    key = (point["group"], point["label"])
    bios_train_point_counts[key] += 1

bios_test_point_counts = defaultdict(int)
for point in bios_test_data:
    key = (point["group"], point["label"])
    bios_test_point_counts[key] += 1


# 提取x, y坐标和大小
x_values = []
y_values = []
sizes = []
for (group, label), count in point_counts.items():
    x_values.append(group)
    y_values.append(label)
    # 你可以根据需要调整大小的比例，例如乘以一个常数
    # sizes.append(count * 5)  # 乘以5是为了让点的大小更明显
    # sizes.append(count * 0.05) # moji
    sizes.append(count * 1) # bios


# 创建散点图
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x_values, y_values, s=sizes, c='blue', alpha=0.6)

# 添加坐标轴和标签
plt.axhline(0.5, color='black', linewidth=0.8)  # 水平线
plt.axvline(0.5, color='black', linewidth=0.8)  # 垂直线
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xticks([0, 1], ["Group 0", "Group 1"])
plt.yticks([0, 1], ["Label 0", "Label 1"])
plt.xlabel("Group")
plt.ylabel("Label")
plt.title("Scatter Plot with Overlapping Points Size Variation")

# 显示网格（可选）
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.show()
plt.savefig('./save_path/fig/bios_data_scatter.png')

print(1)