import cv2
import numpy as np
import os
import sys
from pathlib import Path  # 新增导入
from utils.undistortion import preprocess, undistortion, getRectifyTransform, rectify_image, draw_line, \
    read_calibration_data
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 配置参数（使用Path处理路径）
left_folder = Path(r'/media/junf/零节零壹/picture/0622_0_redTilmpa_rd/左眼(DA5182144)')
right_folder = Path(r'/media/junf/零节零壹/picture/0622_0_redTilmpa_rd/右眼(DA5182138)')
output_base = Path(r'/media/junf/零节零壹/picture/0622_0_redTilmpa_rd/')
calib_file = Path(r'/home/junf/program/MoChaOutputs/uw_photo02.yaml')

# 创建输出目录（使用Path的mkdir方法）
(output_base / 'processing').mkdir(parents=True, exist_ok=True)
(output_base / 'processing' / 'rectified_L').mkdir(parents=True, exist_ok=True)
(output_base / 'processing' / 'rectified_R').mkdir(parents=True, exist_ok=True)
(output_base / 'processing' / 'Combined').mkdir(parents=True, exist_ok=True)

height = 1080
width = 1440

# 读取校准数据
cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T = read_calibration_data(
    calibration_file=str(calib_file))  # 转换为字符串兼容旧代码

# 获取立体校正变换矩阵
map1x, map1y, map2x, map2y, Q, roi1, roi2 = getRectifyTransform(
    height, width, str(calib_file)
)

# 处理同名文件
left_files = set(os.listdir(left_folder))
right_files = set(os.listdir(right_folder))
common_files = left_files.intersection(right_files)


# 检查文件是否已处理（使用Path）
def is_file_processed(filename):
    filename = Path(filename)
    return (
            (output_base / 'rectified_L' / filename).exists() and
            (output_base / 'rectified_R' / filename).exists() and
            (output_base / 'Combined' / filename).exists()
    )


# 处理单个文件（使用Path和文件存在性检查）
def process_file(filename):
    # if not filename.lower().endswith('.png') or is_file_processed(filename):
    if not filename.lower().endswith('.bmp'):
        tqdm.write(f"跳过文件: {filename}")
        return

    try:
        filename = Path(filename)
        left_path = left_folder / filename
        right_path = right_folder / filename

        # 验证文件存在性
        if not left_path.exists() or not right_path.exists():
            tqdm.write(f"文件不存在: {left_path} 或 {right_path}")
            return

        img1 = cv2.imread(str(left_path))  # Path对象转字符串
        img2 = cv2.imread(str(right_path))

        # 检查图像是否成功读取
        if img1 is None or img2 is None:
            tqdm.write(f"无法读取文件: {filename}")
            return

        # CLAHE预处理
        gray1, gray2 = preprocess(img1, img2)

        # 去畸变
        undist1 = undistortion(gray1, cameraMatrix_l, distCoeffs_l)
        undist2 = undistortion(gray2, cameraMatrix_r, distCoeffs_r)

        # 立体校正
        rectified1, rectified2 = rectify_image(undist1, undist2, map1x, map1y, map2x, map2y)
        # rectified1, rectified2, debug_mask = apply_common_mask(rectified1, rectified2, roi1, roi2)

        # # 保存调试掩膜
        # cv2.imwrite(str(output_base / 'debug' / filename), debug_mask)
        # """
        # # 新增——仅保留左右图像交集部分，其余部分为黑色
        # """
        # # 获取 ROI
        # x1, y1, w1, h1 = roi1
        # x2, y2, w2, h2 = roi2
        #
        # # # 计算交集区域
        # # x_start = max(x1, x2)
        # # y_start = max(y1, y2)
        # # x_end = min(x1 + w1, x2 + w2)
        # # y_end = min(y1 + h1, y2 + h2)
        # # w_intersect = max(0, x_end - x_start)
        # # h_intersect = max(0, y_end - y_start)
        # # 计算交集区域（添加边界保护）
        # height, width = rectified1.shape[:2]
        # x_start = max(x1, x2, 0)
        # y_start = max(y1, y2, 0)
        # x_end = min(x1 + w1, x2 + w2, width)
        # w_intersect = max(0, x_end - x_start)
        # h_intersect = max(0, min(h1, h2))
        #
        # # 创建全黑掩膜（保持与输入图像相同的数据类型）
        # mask = np.zeros((height, width), dtype=rectified1.dtype)
        #
        # # 在掩膜上标记交集区域
        # if w_intersect > 0 and h_intersect > 0:
        #     mask[y_start:y_start + h_intersect, x_start:x_start + w_intersect] = 1
        #
        # # 应用掩膜（自动适配单/三通道）
        # rectified1 = cv2.bitwise_and(rectified1, rectified1, mask=mask)
        # rectified2 = cv2.bitwise_and(rectified2, rectified2, mask=mask)
        #
        # # 可选：验证掩膜效果（调试时取消注释）
        # # debug_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 单通道转三通道用于显示
        # # debug_path = output_base / 'debug' / filename
        # # debug_path.parent.mkdir(exist_ok=True)
        # # cv2.imwrite(str(debug_path), debug_mask)
        # # tqdm.write(f"Image Size: {width}x{height}, ROI1: {x1},{y1}+{w1}x{h1}, ROI2: {x2},{y2}+{w2}x{h2}")  # 自动适配多进程环境
        # # tqdm.write(f"Intersection: {x_start}+{w_intersect} (Total Width: {width})")  # 自动适配多进程环境
        #
        # # 将非交集区域设为黑色
        # if w_intersect > 0 and h_intersect > 0:
        #     # 保留交集区域，其他区域置黑
        #     tqdm.write(f"保留交集区域，其他区域置黑")
        #     rectified1[:, :x_start] = 0
        #     rectified1[:, x_start + w_intersect:] = 0
        #     rectified2[:, :x_start] = 0
        #     rectified2[:, x_start + w_intersect:] = 0
        # else:
        #     tqdm.write(f"无交集区域，整张图像设为黑色")
        #     # 无交集区域，整张图像设为黑色
        #     rectified1[:] = 0
        #     rectified2[:] = 0
        # """if w_intersect > 0 and h_intersect > 0:
        #     # 创建掩膜
        #     mask = np.zeros_like(rectified1, dtype=np.uint8)
        #     mask[y_start:y_start + h_intersect, x_start:x_start + w_intersect] = (255, 255, 255)
        #
        #     # 应用掩膜
        #     rectified1 = cv2.bitwise_and(rectified1, mask)
        #     rectified2 = cv2.bitwise_and(rectified2, mask)
        # else:
        #     # 无交集区域，整张图像设为黑色
        #     rectified1[:] = 0
        #     rectified2[:] = 0"""

        # 保存处理后的图像（使用Path）
        rectified1 = cv2.cvtColor(rectified1, cv2.COLOR_GRAY2BGR)  # 添加此行
        rectified2 = cv2.cvtColor(rectified2, cv2.COLOR_GRAY2BGR)  # 添加此行
        cv2.imwrite(str(output_base / 'processing' / 'rectified_L' / f"{filename.stem}.png"), rectified1)
        cv2.imwrite(str(output_base / 'processing' / 'rectified_R' / f"{filename.stem}.png"), rectified2)

        # 拼接左右校正图像
        combined = np.hstack([rectified1, rectified2])
        # combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)  # 单通道转三通道

        # 生成极线叠加层（仅线条）
        overlay = draw_line(rectified1, rectified2)

        # 半透明叠加
        blended = cv2.addWeighted(combined, 1.0, overlay, 0.3, 0)

        # 保存拼接图像
        combined_path = output_base / 'processing' / 'Combined' / filename
        cv2.imwrite(str(combined_path.with_suffix('.png')), blended)
    except Exception as e:
        tqdm.write(f"[{filename}] Error: {str(e)}")  # 捕获并显示异常
        return


# 多进程处理
def main():
    # 过滤未处理的文件
    # files_to_process = [f for f in common_files if not is_file_processed(f)]
    files_to_process = [f for f in common_files]

    # 设置进程池大小
    num_processes = max(1, int(cpu_count() * 0.2))
    # num_processes = 1

    # 使用 tqdm 显示进度条
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process), desc="处理进度", unit="文件"))

    print("处理完成")


if __name__ == "__main__":
    # 单独测试报错文件是否能被 OpenCV 读取
    # test_path = r"G:\Pictures\RedTilmpa\RedTilmpa_0522_0\R\Image_0001747927314995.png"
    # img = cv2.imread(test_path)
    main()
    # files_to_process = [f for f in common_files]
    # process_file(files_to_process[0])
    # print(img is not None)  # 输出 False 则确认文件无法读取
