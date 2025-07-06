import cv2
import os
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm

def natural_sort_key(filename):
    """
    自然排序函数，按照数字大小排序而不是字符串排序
    例如: Image_1.png, Image_2.png, ..., Image_10.png, Image_11.png
    """
    # 提取文件名中的数字部分用于排序
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

def create_video_from_images(input_folder, output_video_path, fps=5):
    """
    将文件夹中的图片按顺序合成视频
    
    参数:
    - input_folder: 输入图片文件夹路径
    - output_video_path: 输出视频路径
    - fps: 视频帧率，默认5
    """
    input_folder = Path(input_folder)
    output_video_path = Path(output_video_path)
    
    # 检查输入文件夹是否存在
    if not input_folder.exists():
        print(f"错误: 输入文件夹不存在 - {input_folder}")
        return False
    
    # 获取所有图片文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    
    for file in input_folder.iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"错误: 在 {input_folder} 中没有找到图片文件")
        return False
    
    # 按文件名（时间戳）排序
    image_files.sort(key=lambda x: natural_sort_key(x.name))
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"第一张: {image_files[0].name}")
    print(f"最后一张: {image_files[-1].name}")
    
    # 读取第一张图片确定视频尺寸
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"错误: 无法读取第一张图片 - {image_files[0]}")
        return False
    
    height, width, channels = first_image.shape
    print(f"图片尺寸: {width}x{height}")
    
    # 创建输出目录
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建视频编写器
    # 使用mp4v编码器，兼容性更好
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("错误: 无法创建视频编写器")
        return False
    
    print(f"开始创建视频，帧率: {fps} fps")
    
    # 逐张图片写入视频
    success_count = 0
    for image_file in tqdm(image_files, desc="生成视频", unit="帧"):
        img = cv2.imread(str(image_file))
        
        if img is None:
            print(f"警告: 无法读取图片 - {image_file}")
            continue
        
        # 确保图片尺寸一致
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
        success_count += 1
    
    # 释放资源
    video_writer.release()
    
    print(f"视频创建完成!")
    print(f"  - 输出路径: {output_video_path}")
    print(f"  - 成功处理: {success_count}/{len(image_files)} 张图片")
    print(f"  - 视频时长: {success_count/fps:.2f} 秒")
    print(f"  - 视频尺寸: {width}x{height}")
    print(f"  - 帧率: {fps} fps")
    
    return True

def main():
    """主函数"""
    # 配置参数
    input_folder = Path(r'/media/junf/零节零壹/picture/0703_6_Huguang/processing/Combined')
    output_folder = Path(r'/media/junf/零节零壹/picture/0703_6_Huguang/processing')
    output_video_name = "combined_stereo_video.mp4"
    fps = 5
    
    output_video_path = output_folder / output_video_name
    
    print("=" * 50)
    print("立体视觉图片转视频工具")
    print("=" * 50)
    print(f"输入目录: {input_folder}")
    print(f"输出视频: {output_video_path}")
    print(f"帧率设置: {fps} fps")
    print("=" * 50)
    
    # 执行转换
    success = create_video_from_images(input_folder, output_video_path, fps)
    
    if success:
        print("\n✅ 视频生成成功!")
        
        # 检查视频文件大小
        if output_video_path.exists():
            file_size = output_video_path.stat().st_size / (1024 * 1024)  # MB
            print(f"视频文件大小: {file_size:.2f} MB")
    else:
        print("\n❌ 视频生成失败!")

if __name__ == "__main__":
    main() 