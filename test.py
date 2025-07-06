from utils.undistortion import *

# %%
# 读取图片
img1 = cv2.imread('L_Image_45.png')
img2 = cv2.imread('R_Image_45.png')
height = img1.shape[0]
width = img1.shape[1]
print(f"height: {height}")
print(f"width: {width}")
print(f"原始图像 shape: {img1.shape}")
# %%
# 获取相机内参和畸变系数
cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T = read_calibration_data(
    calibration_file='uw_photo01_3k.yaml')
# 获取立体校正变换矩阵
map1x, map1y, map2x, map2y, Q = getRectifyTransform(
    height, width,
    'uw_photo01_3k.yaml')
print(f"Q: {Q}")

# 输出原图
line_img_origin = draw_line(img1, img2)
cv2.imwrite('./outputs_2/uw_line_img_origin.png', line_img_origin)
print(f"已输出line_img_origin.png")

# %%
# 第一次畸变矫正和立体校正
# 消除畸变
rectified_img1_ = undistortion(img1, cameraMatrix_l, distCoeffs_l)
rectified_img2_ = undistortion(img2, cameraMatrix_r, distCoeffs_r)
line_img = draw_line(rectified_img1_, rectified_img2_)
cv2.imwrite('./outputs_2/uw_line_img_0.png', line_img)
print(f"已输出line_img_0.png")
# 立体校正
rectified_img1_, rectified_img2_ = rectify_image(rectified_img1_, rectified_img2_, map1x, map1y, map2x, map2y)

# rectified_img1_, rectified_img2_ = preprocess(rectified_img1_, rectified_img2_)  # 消除光照不均,可要可不要

# 立体校正检验(画线) 在调用 draw_line 之前, 将单通道灰度图转为三通道图像
# rectified_img1_color = cv2.cvtColor(rectified_img1_, cv2.COLOR_GRAY2BGR)
# rectified_img2_color = cv2.cvtColor(rectified_img2_, cv2.COLOR_GRAY2BGR)
# line_img = draw_line(rectified_img1_color, rectified_img2_color)
line_img = draw_line(rectified_img1_, rectified_img2_)  # 没有进行CLAHE增强
cv2.imwrite('./outputs_2/uw_line_img_1.png', line_img)
print(f"已输出line_img_1.png")
# cv2.imwrite('./outputs/uw_rectified_img1_color.png', rectified_img1_color)
# cv2.imwrite('./outputs/uw_rectified_img2_color.png', rectified_img2_color)
cv2.imwrite('./outputs_2/uw_rectified_img1_color.png', rectified_img1_)  # 没有进行CLAHE增强
cv2.imwrite('./outputs_2/uw_rectified_img2_color.png', rectified_img2_)  # 没有进行CLAHE增强

