import os
import shutil
from tqdm import tqdm  # 导入进度条库

# 定义源文件夹和目标文件夹
left_source = r'G:\Pictures\赛尔号\红罗非鱼_0504_0\左眼(DA5182144)'
right_source = r'G:\Pictures\赛尔号\红罗非鱼_0504_0\右眼(DA5182138)'
left_dest = r'C:\RedTilmpa\0504_0\L'
right_dest = r'C:\RedTilmpa\0504_0\R'

# 确保目标文件夹存在
os.makedirs(left_dest, exist_ok=True)
os.makedirs(right_dest, exist_ok=True)

# 获取两个文件夹中的文件列表
left_files = set(os.listdir(left_source))
right_files = set(os.listdir(right_source))

# 找到两个文件夹中命名相同的文件
common_files = left_files.intersection(right_files)
print(f"共有 {len(common_files)} 个文件命名相同。")

# 使用 tqdm 包装循环并显示进度条
for file_name in tqdm(common_files, total=len(common_files)):  # total 参数保证进度条准确性
    shutil.copy(os.path.join(left_source, file_name), os.path.join(left_dest, file_name))
    shutil.copy(os.path.join(right_source, file_name), os.path.join(right_dest, file_name))

print(f"已复制 {len(common_files)} 个文件到目标文件夹。")

# # 复制这些文件到目标文件夹
# for file_name in common_files:
#     shutil.copy(os.path.join(left_source, file_name), os.path.join(left_dest, file_name))
#     shutil.copy(os.path.join(right_source, file_name), os.path.join(right_dest, file_name))
#     print(f"已复制 {file_name} 到目标文件夹。")
# print(f"已复制 {len(common_files)} 个文件到目标文件夹。")
