import cv2
import argparse

def load_image(image_path):
    """加载图像并进行有效性检查"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}，请检查路径是否正确")
    return image

def apply_filter(image, filter_type, kernel_size=3, sigma=0.5, d=5, sigma_color=50, sigma_space=50):
    """
    应用滤波处理（新增参数微调功能，降低去噪强度）
    参数说明：
    - kernel_size: 卷积核大小（关键！越小去噪越弱、保细节越好，需为奇数）
    - sigma: 高斯滤波的标准差（越小平滑越弱）
    - d: 双边滤波的邻域直径（越小局部平滑范围越小）
    - sigma_color: 双边滤波的颜色相似度系数（越小对颜色差异越敏感，保细节越好）
    - sigma_space: 双边滤波的空间距离系数（越小对空间距离越敏感，保细节越好）
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if filter_type == "gaussian":
        # 优化点：默认核从5×5→3×3，sigma从自动→0.5（降低平滑强度）
        filtered = cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigma)
    
    elif filter_type == "median":
        # 优化点：默认核从5×5→3×3（中值滤波核越小，细节保留越多）
        filtered = cv2.medianBlur(image_rgb, kernel_size)
    
    elif filter_type == "bilateral":
        # 优化点：默认直径从9→5，颜色/空间系数从75→50（减弱保边平滑强度）
        filtered = cv2.bilateralFilter(image_rgb, d, sigma_color, sigma_space)
    
    elif filter_type == "average":
        # 优化点：默认核从5×5→3×3（均值滤波核越小，模糊越轻）
        filtered = cv2.blur(image_rgb, (kernel_size, kernel_size))
    
    else:
        raise ValueError(f"不支持的滤波类型: {filter_type}，可选类型：gaussian, median, bilateral, average")
    
    return cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)

def save_image(image, output_path):
    """保存处理后的图像"""
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"无法保存图像到: {output_path}，请检查路径是否可写")
    print(f"处理后的图像已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='图像滤波处理工具（支持去噪强度微调）')
    # 基础参数（保留原功能）
    parser.add_argument('--input', required=True, help='输入图像路径')
    parser.add_argument('--output', required=True, help='输出图像路径')
    parser.add_argument('--filter', choices=['gaussian', 'median', 'bilateral', 'average'], 
                      default='bilateral', help='滤波类型，默认：bilateral（保边效果好，不易过糊）')
    
    # 新增：去噪强度微调参数（均设默认值，不填则用弱去噪配置）
    parser.add_argument('--kernel_size', type=int, default=3, help='卷积核大小（奇数，3/5/7，默认3→弱去噪）')
    parser.add_argument('--sigma', type=float, default=0.5, help='仅高斯滤波：标准差（0.1-2.0，默认0.5→弱平滑）')
    parser.add_argument('--d', type=int, default=5, help='仅双边滤波：邻域直径（3/5/7/9，默认5→弱保边平滑）')
    parser.add_argument('--sigma_color', type=int, default=50, help='仅双边滤波：颜色相似度系数（10-100，默认50→保细节）')
    parser.add_argument('--sigma_space', type=int, default=50, help='仅双边滤波：空间距离系数（10-100，默认50→保细节）')
    
    args = parser.parse_args()
    
    # 校验：确保卷积核为奇数（避免OpenCV报错）
    if args.kernel_size % 2 == 0:
        args.kernel_size += 1  # 若输入偶数，自动+1转为奇数
        print(f"提示：卷积核需为奇数，已自动调整为 {args.kernel_size}")
    
    try:
        image = load_image(args.input)
        print(f"成功加载图像: {args.input}，当前去噪强度：弱（可通过参数调整）")
        
        # 传入微调参数，应用滤波
        filtered_image = apply_filter(
            image=image,
            filter_type=args.filter,
            kernel_size=args.kernel_size,
            sigma=args.sigma,
            d=args.d,
            sigma_color=args.sigma_color,
            sigma_space=args.sigma_space
        )
        
        save_image(filtered_image, args.output)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()
