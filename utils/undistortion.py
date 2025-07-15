import cv2
import numpy as np
import yaml  # 导入相机标定的参数


# 1. 灰度预处理
# %%
def preprocess(img1, img2) -> tuple:
    # 彩色图->灰度图
    if img1.ndim == 3:  # 判断为三维数组
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # # 直方图均衡
    # img1 = cv2.equalizeHist(img1)
    # img2 = cv2.equalizeHist(img2)

    # CLAHE
    '''
    Args:
        - clipLimit (float): 从全局对比度的角度出发，对比度受限的程度。默认为 2.0。当 clipLimit 设置为 0 或者负值时，表示没有对比度限制。较高的值会增加对比度，但可能导致噪声放大。
        - tileGridSize (tuple of two ints): 每个小网格的大小，以像素为单位（行数，列数）。默认值为 (8, 8)。图像将被分为多个大小相同的网格块，CLAHE 算法分别对每个网格块进行直方图均衡化。
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)

    cv2.CLAHE.collectGarbage(clahe)
    return img1, img2


# 彩色图像预处理 - 保留颜色的同时应用CLAHE
def preprocess_color(img1, img2) -> tuple:
    """
    对彩色图像应用CLAHE增强，同时保留原有颜色信息
    在HSV色彩空间中只对V（亮度）通道应用CLAHE，保留H（色调）和S（饱和度）
    
    Args:
        img1, img2: 输入的彩色图像 (BGR格式)
    
    Returns:
        tuple: 处理后的彩色图像 (BGR格式)
    """
    def apply_clahe_color(img):
        if img.ndim == 2:  # 如果是灰度图，直接应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(img)
            cv2.CLAHE.collectGarbage(clahe)
            return result
        
        # 转换到HSV色彩空间,H:色调(Hue),S:饱和度(Saturation),V:亮度(Value)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 只对V（亮度）通道应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        cv2.CLAHE.collectGarbage(clahe)
        
        # 合并通道并转换回BGR
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        return bgr_enhanced
    
    # 对两张图像分别应用CLAHE增强
    img1_enhanced = apply_clahe_color(img1)
    img2_enhanced = apply_clahe_color(img2)
    
    return img1_enhanced, img2_enhanced


# %%
# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# %% md
# 2. 获取畸变矫正和立体校正变换矩阵/重投影矩阵
# %%
def read_calibration_data(calibration_file):
    with open(calibration_file, 'r') as f:
        calib_data = yaml.safe_load(f)
        cameraMatrix_l = np.array(calib_data['camera_matrix_left']['data']).reshape(3, 3)
        distCoeffs_l = np.array(calib_data['dist_coeff_left']['data'])
        cameraMatrix_r = np.array(calib_data['camera_matrix_right']['data']).reshape(3, 3)
        distCoeffs_r = np.array(calib_data['dist_coeff_right']['data'])
        R = np.array(calib_data['R']['data']).reshape(3, 3)
        T = np.array(calib_data['T']['data']).reshape(3, 1)
    return cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T


# %%
# 立体校正算法
def getRectifyTransform(height, width, calibration_file):
    cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T = read_calibration_data(calibration_file)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r,
                                                      (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix_l, distCoeffs_l, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix_r, distCoeffs_r, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q, roi1, roi2

# #%%
# # 交集区域计算函数
# def get_intersection_roi(roi1, roi2, img_shape):
#     x1, y1, w1, h1 = roi1
#     x2, y2, w2, h2 = roi2
#     height, width = img_shape[:2]
#
#     # 计算交集区域（添加边界保护）
#     x_start = max(x1, x2, 0)
#     y_start = max(y1, y2, 0)
#     x_end = min(x1 + w1, x2 + w2, width)
#     w_intersect = max(0, x_end - x_start)
#     h_intersect = max(0, min(y1 + h1, y2 + h2) - y_start)
#
#     return x_start, y_start, w_intersect, h_intersect
# #%%
# # 掩码生成与应用
# def apply_common_mask(rectified1, rectified2, roi1, roi2):
#     # 获取图像尺寸
#     height, width = rectified1.shape[:2]
#
#     # 计算交集区域
#     x, y, w, h = get_intersection_roi(roi1, roi2, rectified1.shape)
#
#     # 创建掩膜（自动适配单/三通道）
#     if len(rectified1.shape) == 3:
#         mask = np.zeros((height, width, 1), dtype=np.uint8)
#     else:
#         mask = np.zeros((height, width), dtype=np.uint8)
#
#     # 填充交集区域
#     if w > 0 and h > 0:
#         mask[y:y + h, x:x + w] = 255
#
#     # 创建输出掩膜（包含调试信息）
#     debug_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(rectified1.shape) == 3 else mask.copy()
#
#     # 应用掩膜
#     masked1 = cv2.bitwise_and(rectified1, rectified1, mask=mask)
#     masked2 = cv2.bitwise_and(rectified2, rectified2, mask=mask)
#
#     return masked1, masked2, debug_mask


# %% md
# 3. 畸变矫正和立体校正
# %%
def rectify_image(img1, img2, map1x, map1y, map2x, map2y):
    rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LANCZOS4)
    rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LANCZOS4)
    # cv2.INTER_LINEAR双线性插值 或 cv2.INTER_LANCZOS4Lanczos插值或cv2.INTER_CUBIC双三次插值
    return rectified_img1, rectified_img2


# %% md
# 4. 立体校正检验----画线
# %%
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    
    # 创建三通道BGR图像用于绘制半透明线条
    output = np.zeros((height, width, 3), dtype=np.uint8)
    # output[0:image1.shape[0], 0:image1.shape[1]] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    # output[0:image2.shape[0], image1.shape[1]:] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    # 检查image1是否是单通道，若是则转换为三通道
    if len(image1.shape) == 2:
        output[0:image1.shape[0], 0:image1.shape[1]] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    else:
        output[0:image1.shape[0], 0:image1.shape[1]] = image1

    # 同样处理image2
    if len(image2.shape) == 2:
        output[0:image2.shape[0], image1.shape[1]:] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    else:
        output[0:image2.shape[0], image1.shape[1]:] = image2
    
    # 绘制半透明平行线
    line_interval = 50
    for k in range(height // line_interval):
        y = line_interval * (k + 1)
        # 使用cv2.LINE_AA抗锯齿，颜色值保持为白色，透明度通过叠加控制
        cv2.line(output, (0, y), (width, y), (255, 255, 255),
                 thickness=2,
                 lineType=cv2.LINE_AA)

    return output
