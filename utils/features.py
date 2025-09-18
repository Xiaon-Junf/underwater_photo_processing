import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import entropy

def calculate_feature_count(image_path):
    """
    计算水下图像的多特征数量
    包括：边缘特征、角点特征、纹理特征（改进版）
    """
    # 1. 读取图像并预处理
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为灰度图用于特征提取
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 去噪预处理，适合水下图像
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 提取边缘特征（Canny边缘检测）
    edges = cv2.Canny(denoised, threshold1=30, threshold2=100)
    edge_count = np.sum(edges > 0)  # 统计边缘像素点数量
    
    # 3. 提取角点特征（Harris角点检测）
    harris_corners = cv2.cornerHarris(denoised, blockSize=3, ksize=3, k=0.04)
    # 阈值化处理，筛选出强角点
    corner_threshold = 0.01 * harris_corners.max()
    corner_count = np.sum(harris_corners > corner_threshold)
    
    # 4. 提取纹理特征（改进的LBP方法）
    radius = 3
    n_points = 8 * radius
    # 使用默认方法计算LBP，获取更多模式变化
    lbp = local_binary_pattern(denoised, n_points, radius, method='default')
    
    # 计算LBP直方图，反映纹理分布
    # 使用64个bin来量化不同的纹理模式
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, n_points + 2))
    
    # 从直方图中提取统计特征，反映纹理变化
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)  # 归一化
    
    # 计算纹理特征：直方图的熵（反映纹理复杂度）
    texture_entropy = entropy(hist)
    
    # 计算纹理特征：直方图的方差（反映纹理分布均匀性）
    texture_variance = np.var(hist)
    
    # 计算纹理特征：高响应 bins 的数量（反映显著纹理模式数量）
    significant_bins = np.sum(hist > 0.01)  # 阈值可调整
    
    # 综合纹理特征，加权组合以匹配其他特征数量级
    texture_count = int((texture_entropy * 100) + (texture_variance * 5000) + (significant_bins * 50))
    
    # 5. 计算总特征数量
    total_count = edge_count + corner_count + texture_count
    
    # 返回各特征数量及总和
    return {
        "边缘特征数量": edge_count,
        "角点特征数量": corner_count,
        "纹理特征数量": texture_count,
        "纹理熵（复杂度）": round(texture_entropy, 2),
        "纹理方差（分布均匀性）": round(texture_variance, 6),
        "显著纹理模式数": significant_bins,
        "总特征数量": total_count
    }

if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = "clahenocolor+dark+fresult.jpg"
    
    try:
        feature_stats = calculate_feature_count(image_path)
        
        # 打印统计结果
        print("水下图像特征数量统计结果：")
        for feature, count in feature_stats.items():
            print(f"{feature}: {count}")
            
    except Exception as e:
        print(f"处理出错: {str(e)}")
