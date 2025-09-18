import cv2
import numpy as np

def dark_channel_prior(image, patch_size=15):
    b, g, r = cv2.split(image)
    dark = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(dark, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, percent=0.001):
    h, w = image.shape[:2]
    flat_dark = dark_channel.ravel()
    num_pixels = h * w
    num_bright = int(max(num_pixels * percent, 1))
    indices = np.argsort(flat_dark)[-num_bright:]
    atmospheric_light = np.zeros(3)
    for i in range(3):
        channel = image[:, :, i].ravel()
        atmospheric_light[i] = np.mean(channel[indices])
    return atmospheric_light

def transmission_estimation(image, atmospheric_light, omega=0.95, patch_size=15):
    h, w = image.shape[:2]  # 获取图像高宽
    normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        normalized[:, :, i] = image[:, :, i] / atmospheric_light[i]
    trans_dark = dark_channel_prior(normalized, patch_size)
    # 关键修改1：生成透射率后，立即增加通道维度（变成(h,w,1)）
    transmission = (1 - omega * trans_dark)[:, :, np.newaxis]
    return transmission

def guided_filter(image, p, r=60, eps=0.001):
    # 关键修改2：确保输入的p是2维（导向滤波要求输入为单通道）
    if len(p.shape) == 3:
        p = p.squeeze(axis=2)  # 若p是(h,w,1)，压缩为(h,w)
    mean_i = cv2.boxFilter(image, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_ip = cv2.boxFilter(image * p, cv2.CV_64F, (r, r))
    cov_ip = mean_ip - mean_i * mean_p
    mean_ii = cv2.boxFilter(image * image, cv2.CV_64F, (r, r))
    var_i = mean_ii - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 关键修改3：滤波后恢复为(h,w,1)，保持维度一致
    filtered = (mean_a * image + mean_b)[:, :, np.newaxis]
    return filtered

def recover_image(image, atmospheric_light, transmission, t0=0.1):
    # 此时transmission已是(h,w,1)，无需再调整维度
    transmission = np.maximum(transmission, t0)  # 直接比较，无需新增维度
    recovered = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        # 图像通道(i)是(h,w)，transmission是(h,w,1)，可兼容广播
        recovered[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission.squeeze(axis=2) + atmospheric_light[i]
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    return recovered

def dark_channel_enhance(input_path, output_path, patch_size=15, omega=0.95, t0=0.1):
    # 读取图像
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"无法读取图像: {input_path}")
    h, w = image.shape[:2]
    print(f"图像尺寸: {w}x{h}")  # 验证图像是否正常加载
    
    # 转换为浮点型（避免整数运算溢出）
    image_float = image.astype(np.float32)
    
    # 1. 计算暗通道（2维：h,w）
    dark = dark_channel_prior(image)
    # 2. 估计大气光（1维：3）
    a = estimate_atmospheric_light(image_float, dark)
    # 3. 估计初始透射率（3维：h,w,1）
    t = transmission_estimation(image_float, a, omega, patch_size)
    print(f"初始透射率形状: {t.shape}")  # 应输出 (288,421,1)
    # 4. 导向滤波优化透射率（输入输出均为3维：h,w,1）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0  # 2维：h,w
    t_refined = guided_filter(gray, t)
    print(f"优化后透射率形状: {t_refined.shape}")  # 应输出 (288,421,1)
    # 5. 恢复图像
    enhanced = recover_image(image_float, a, t_refined, t0)
    
    # 保存结果
    cv2.imwrite(output_path, enhanced)
    print(f"增强图像已保存至: {output_path}")
    return enhanced

if __name__ == "__main__":
    input_path = r"D:\code-py\3d\fresult.jpg"
    output_path = r"D:\code-py\3d\dark+fresult.jpg"
    try:
        dark_channel_enhance(input_path, output_path)
    except Exception as e:
        print(f"处理失败: {str(e)}")
