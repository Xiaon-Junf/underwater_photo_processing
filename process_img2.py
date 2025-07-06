import cv2
import numpy as np
import os
from utils.undistortion import preprocess, undistortion, getRectifyTransform, rectify_image, draw_line, \
    read_calibration_data
from tqdm import tqdm  # 导入进度条库

# 配置参数
left_folder = r'C:\RedTilmpa\0502_0\L'
right_folder = r'C:\RedTilmpa\0502_0\R'
output_base = r'C:\RedTilmpa\0502_0\processing'
calib_file = r'D:\开心摸鱼项目\MoChaOutputs\uw_photo02.yaml'

# 创建输出目录
os.makedirs(os.path.join(output_base, 'rectified_L'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'rectified_R'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'Combined'), exist_ok=True)

height = 1080
width = 1440

# 读取校准数据
cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T = read_calibration_data(
    calibration_file=calib_file)
# 获取立体校正变换矩阵
map1x, map1y, map2x, map2y, Q = getRectifyTransform(
    height, width, calib_file
)

# 处理同名文件
left_files = set(os.listdir(left_folder))
right_files = set(os.listdir(right_folder))
common_files = left_files.intersection(right_files)

# 使用 tqdm 包装循环并显示进度条
for filename in tqdm(common_files, total=len(common_files), desc="处理进度", unit="文件"):
    if not filename.lower().endswith('.png'):
        continue

    # 读取图像
    left_path = os.path.join(left_folder, filename)
    right_path = os.path.join(right_folder, filename)
    img1 = cv2.imread(left_path)
    img2 = cv2.imread(right_path)

    # CLAHE预处理
    gray1, gray2 = preprocess(img1, img2)

    # 去畸变
    undist1 = undistortion(gray1, cameraMatrix_l, distCoeffs_l)
    undist2 = undistortion(gray2, cameraMatrix_r, distCoeffs_r)

    # 立体校正
    rectified1, rectified2 = rectify_image(undist1, undist2, map1x, map1y, map2x, map2y)

    # 保存处理后的图像
    cv2.imwrite(os.path.join(output_base, 'rectified_L', filename), rectified1)
    cv2.imwrite(os.path.join(output_base, 'rectified_R', filename), rectified2)

    # 拼接左右校正图像
    combined = np.hstack([rectified1, rectified2])
    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)  # 单通道转三通道

    # 生成极线叠加层（仅线条）
    overlay = draw_line(rectified1, rectified2)

    # 半透明叠加
    blended = cv2.addWeighted(combined, 1.0, overlay, 0.3, 0)

    # 保存拼接图像
    combined_path = os.path.join(output_base, 'Combined', filename)
    cv2.imwrite(combined_path, blended)

print("处理完成")
