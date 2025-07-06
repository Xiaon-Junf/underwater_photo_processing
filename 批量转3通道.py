import cv2
import os
from pathlib import Path
from tqdm import tqdm

def convert_to_3channel(image_path):
    """强制转为3通道 RGB 格式，并去除 alpha 通道"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"⚠️ 无法读取图像: {image_path.name}")
        return False
    
    # 处理单通道灰度图
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # 处理 alpha 通道（4通道 -> 3通道）
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    
    # 确保最终是3通道
    if img.ndim != 3 or img.shape[2] != 3:
        print(f"❌ 转换失败: {image_path.name} -> shape={img.shape}")
        return False
    
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"✅ 进行BGR -> RGB: {image_path.name} -> shape={img.shape}")
        
    # 强制覆盖保存为无 alpha 的 PNG
    cv2.imwrite(str(image_path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return True

# 定义路径
left_folder = Path(r"/media/junf/C6769873769865C9/RedTilmpa/0509_0/processing/rectified_L")
right_folder = Path(r"/media/junf/C6769873769865C9/RedTilmpa/0509_0/processing/rectified_R")

all_files = list(left_folder.glob("*.png")) + list(right_folder.glob("*.png"))

# 执行转换并输出日志
print("🔄 正在强制转换为3通道...")
converted_count = 0
for file_path in tqdm(all_files, total=len(all_files), desc="转换进度", unit="文件"):
    if convert_to_3channel(file_path):
        converted_count += 1

print(f"✅ 已转换 {converted_count}/{len(all_files)} 个文件")

# 验证转换结果
def verify_conversion(folder):
    for img_file in Path(folder).glob("*.png"):
        img = cv2.imread(str(img_file))
        if img.ndim != 3 or img.shape[2] != 3:
            print(f"❌ 异常文件: {img_file.name} -> shape={img.shape}")
        else:
            print(f"✅ 验证通过: {img_file.name} -> shape={img.shape}")

print("🔍 正在验证转换结果...")
verify_conversion(left_folder)
verify_conversion(right_folder)