import cv2
import numpy as np
import os
import sys
from pathlib import Path
from utils.undistortion import preprocess, undistortion, getRectifyTransform, rectify_image, draw_line, \
    read_calibration_data
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# 批量处理配置
base_path = Path(r'/media/junf/零节零壹/picture')
# 定义要处理的7个目录
directories_to_process = [
    '0703_0_Huguang',
    '0703_1_Huguang', 
    '0703_2_Huguang',
    '0703_3_Huguang',
    '0703_4_Huguang',
    '0703_5_Huguang',
    '0703_6_Huguang'
]

# 校准文件路径
calib_file = Path(r'/home/junf/program/MoChaOutputs/uw_photo02.yaml')

# 图像尺寸参数
height = 1080
width = 1440

# 读取校准数据
cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T = read_calibration_data(
    calibration_file=str(calib_file))

# 获取立体校正变换矩阵
map1x, map1y, map2x, map2y, Q, roi1, roi2 = getRectifyTransform(
    height, width, str(calib_file)
)

def setup_directories(output_base):
    """为单个目录创建输出文件夹结构"""
    (output_base / 'processing').mkdir(parents=True, exist_ok=True)
    (output_base / 'processing' / 'rectified_L').mkdir(parents=True, exist_ok=True)
    (output_base / 'processing' / 'rectified_R').mkdir(parents=True, exist_ok=True)
    (output_base / 'processing' / 'Combined').mkdir(parents=True, exist_ok=True)

def get_common_files(left_folder, right_folder):
    """获取左右文件夹中的共同文件"""
    if not left_folder.exists() or not right_folder.exists():
        print(f"警告: 文件夹不存在 - {left_folder} 或 {right_folder}")
        return set()
    
    left_files = set(os.listdir(left_folder))
    right_files = set(os.listdir(right_folder))
    return left_files.intersection(right_files)

def process_file(args):
    """处理单个文件"""
    filename, left_folder, right_folder, output_base = args
    
    if not filename.lower().endswith(('.bmp', '.png')):
        return f"跳过文件: {filename}"

    try:
        filename = Path(filename)
        left_path = left_folder / filename
        right_path = right_folder / filename

        # 验证文件存在性
        if not left_path.exists() or not right_path.exists():
            return f"文件不存在: {left_path} 或 {right_path}"

        img1 = cv2.imread(str(left_path))
        img2 = cv2.imread(str(right_path))

        # 检查图像是否成功读取
        if img1 is None or img2 is None:
            return f"无法读取文件: {filename}"

        # CLAHE预处理
        gray1, gray2 = preprocess(img1, img2)

        # 去畸变
        undist1 = undistortion(gray1, cameraMatrix_l, distCoeffs_l)
        undist2 = undistortion(gray2, cameraMatrix_r, distCoeffs_r)

        # 立体校正
        rectified1, rectified2 = rectify_image(undist1, undist2, map1x, map1y, map2x, map2y)

        # 转换为三通道图像
        rectified1 = cv2.cvtColor(rectified1, cv2.COLOR_GRAY2BGR)
        rectified2 = cv2.cvtColor(rectified2, cv2.COLOR_GRAY2BGR)
        
        # 保存处理后的图像
        cv2.imwrite(str(output_base / 'processing' / 'rectified_L' / f"{filename.stem}.png"), rectified1)
        cv2.imwrite(str(output_base / 'processing' / 'rectified_R' / f"{filename.stem}.png"), rectified2)

        # 拼接左右校正图像
        combined = np.hstack([rectified1, rectified2])

        # 生成极线叠加层
        overlay = draw_line(rectified1, rectified2)

        # 半透明叠加
        blended = cv2.addWeighted(combined, 1.0, overlay, 0.3, 0)

        # 保存拼接图像
        combined_path = output_base / 'processing' / 'Combined' / filename
        cv2.imwrite(str(combined_path.with_suffix('.png')), blended)
        
        return f"成功处理: {filename}"
        
    except Exception as e:
        return f"[{filename}] Error: {str(e)}"

def process_single_directory(directory_name):
    """处理单个目录的所有文件"""
    print(f"\n开始处理目录: {directory_name}")
    
    # 设置路径
    base_dir = base_path / directory_name
    left_folder = base_dir / '左眼(DA5182144)'
    right_folder = base_dir / '右眼(DA5182138)'
    output_base = base_dir
    
    # 创建输出目录
    setup_directories(output_base)
    
    # 获取共同文件
    common_files = get_common_files(left_folder, right_folder)
    if not common_files:
        print(f"警告: {directory_name} 中没有找到共同文件")
        return
    
    print(f"找到 {len(common_files)} 个共同文件")
    
    # 准备处理参数
    file_args = [(filename, left_folder, right_folder, output_base) for filename in common_files]
    
    # 设置进程池大小
    num_processes = max(1, int(cpu_count() * 0.4))
    
    # 处理文件
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file, file_args), 
            total=len(file_args), 
            desc=f"处理 {directory_name}", 
            unit="文件"
        ))
    
    end_time = time.time()
    
    # 统计结果
    success_count = sum(1 for result in results if "成功处理" in result)
    error_count = len(results) - success_count
    
    print(f"{directory_name} 处理完成:")
    print(f"  - 成功: {success_count} 个文件")
    print(f"  - 失败: {error_count} 个文件")
    print(f"  - 耗时: {end_time - start_time:.2f} 秒")
    
    # 显示错误信息
    if error_count > 0:
        print("错误详情:")
        for result in results:
            if "成功处理" not in result:
                print(f"  - {result}")

def main():
    """主函数 - 批量处理所有目录"""
    print("开始批量处理立体校正...")
    print(f"待处理目录数量: {len(directories_to_process)}")
    print(f"校准文件: {calib_file}")
    
    total_start_time = time.time()
    
    # 逐个处理目录
    for i, directory_name in enumerate(directories_to_process, 1):
        print(f"\n{'='*50}")
        print(f"进度: {i}/{len(directories_to_process)}")
        process_single_directory(directory_name)
    
    total_end_time = time.time()
    
    print(f"\n{'='*50}")
    print("全部处理完成!")
    print(f"总耗时: {total_end_time - total_start_time:.2f} 秒")
    print(f"平均每个目录: {(total_end_time - total_start_time)/len(directories_to_process):.2f} 秒")

if __name__ == "__main__":
    main() 