import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm  # 导入tqdm

left_eye_folder = r'D:\Pictures\湖光岩_0\左眼 (DA5182144)'
right_eye_folder = r'D:\Pictures\湖光岩_0\右眼 (DA5182138)'


# 提取时间戳
def extract_timestamp(filename):
    return int(filename.split('_')[1].split('.')[0])


# 获取文件名列表
print("正在获取文件名列表...")
left_eye_filenames = os.listdir(left_eye_folder)
right_eye_filenames = os.listdir(right_eye_folder)

# 提取时间戳（添加进度条）
print("\n正在提取时间戳...")
left_eye_timestamps = [extract_timestamp(filename) for filename in tqdm(left_eye_filenames) if
                       filename.endswith('.png')]
right_eye_timestamps = [extract_timestamp(filename) for filename in tqdm(right_eye_filenames) if
                        filename.endswith('.png')]

# 将时间戳转换为numpy数组
left_times = np.array(left_eye_timestamps).reshape(-1, 1)
right_times = np.array(right_eye_timestamps).reshape(-1, 1)

# 计算时间戳之间的距离（差异）
print("\n正在计算时间戳差异...")
distances = cdist(left_times, right_times, 'euclidean')

# 找到最小距离的索引
min_index = np.unravel_index(np.argmin(distances), distances.shape)

# 获取对应的图像对
left_index = min_index[0]
right_index = min_index[1]

left_timestamp = left_eye_timestamps[left_index]
right_timestamp = right_eye_timestamps[right_index]

print(f"\n最同步的图像对：左眼图像时间戳 {left_timestamp}，右眼图像时间戳 {right_timestamp}")
print(f"时间戳差异：{distances[left_index, right_index]} 微秒")

# 打印所有时间戳差异（添加进度条）
print("\n正在计算所有时间戳差异...")
all_time_differences = []
for i, left_time in tqdm(enumerate(left_eye_timestamps), total=len(left_eye_timestamps)):
    for j, right_time in enumerate(right_eye_timestamps):
        diff = abs(left_time - right_time)
        all_time_differences.append(diff)

# 设置同步阈值
sync_threshold = 5

# 找出所有同步的图像对（添加进度条）
print("\n正在查找同步图像对...")
sync_pairs = []
for i, left_time in tqdm(enumerate(left_eye_timestamps), total=len(left_eye_timestamps)):
    for j, right_time in enumerate(right_eye_timestamps):
        diff = abs(left_time - right_time)
        if diff <= sync_threshold:
            sync_pairs.append((i, j))

print(f"\n找到的同步图像对数量：{len(sync_pairs)}")

# 创建新的文件夹来保存同步的图像对
sync_left_eye_folder = r'D:\Pictures\Huguangyan\L'
sync_right_eye_folder = r'D:\Pictures\Huguangyan\R'

if not os.path.exists(sync_left_eye_folder):
    os.makedirs(sync_left_eye_folder)
if not os.path.exists(sync_right_eye_folder):
    os.makedirs(sync_right_eye_folder)

# 保存同步的图像对（添加进度条）
print("\n正在保存同步图像对...")
for left_index_sync, right_index_sync in tqdm(sync_pairs, desc="复制文件中"):
    left_filename = left_eye_filenames[left_index_sync]
    right_filename = right_eye_filenames[right_index_sync]

    left_filepath = os.path.join(left_eye_folder, left_filename)
    right_filepath = os.path.join(right_eye_folder, right_filename)

    sync_left_filepath = os.path.join(sync_left_eye_folder, left_filename)
    sync_right_filepath = os.path.join(sync_right_eye_folder, right_filename)

    with open(left_filepath, 'rb') as src, open(sync_left_filepath, 'wb') as dst:
        dst.write(src.read())

    with open(right_filepath, 'rb') as src, open(sync_right_filepath, 'wb') as dst:
        dst.write(src.read())

print(f"\n同步图像对已保存到 {sync_left_eye_folder} 和 {sync_right_eye_folder}")

# 打印所有时间戳差异的分布情况
print(f"\n所有时间戳差异的最大值：{max(all_time_differences)} 微秒")
print(f"所有时间戳差异的最小值：{min(all_time_differences)} 微秒")
