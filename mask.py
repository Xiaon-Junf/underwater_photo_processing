import cv2
import numpy as np
import yaml


def read_calibration_data(calibration_file):
    """从YAML文件读取双目相机标定参数"""
    with open(calibration_file, 'r') as f:
        calib_data = yaml.safe_load(f)
        cameraMatrix_l = np.array(calib_data['camera_matrix_left']['data']).reshape(3, 3)
        distCoeffs_l = np.array(calib_data['dist_coeff_left']['data'])
        cameraMatrix_r = np.array(calib_data['camera_matrix_right']['data']).reshape(3, 3)
        distCoeffs_r = np.array(calib_data['dist_coeff_right']['data'])
        R = np.array(calib_data['R']['data']).reshape(3, 3)
        T = np.array(calib_data['T']['data']).reshape(3, 1)
    return cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T


def create_stereo_masks(imgL, imgR, calib_data_path):
    """
    生成双目图像重叠区域的精确掩码（主函数）
    参数:
        imgL, imgR: 左右图像 (H,W,C)
        calib_data_path: YAML标定文件路径
    返回:
        maskedL, maskedR: 掩码处理后的图像
        maskL, maskR: 生成的掩码 (可用于可视化)
    """
    # 1. 读取标定参数
    K1, D1, K2, D2, R, T = read_calibration_data(calib_data_path)
    h, w = imgL.shape[:2]
    image_size = (w, h)

    # 2. 立体校正计算
    R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # 保持主点不变
    )

    # 3. 生成映射表（实际用于校正图像）
    mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    # 4. 创建初始掩码（全黑）
    maskL = np.zeros((h, w), dtype=np.uint8)
    maskR = np.zeros((h, w), dtype=np.uint8)

    # 5. 在掩码上绘制有效区域（使用校正后的ROI）
    cv2.rectangle(maskL, (roiL[0], roiL[1]), (roiL[0] + roiL[2], roiL[1] + roiL[3]), 255, -1)
    cv2.rectangle(maskR, (roiR[0], roiR[1]), (roiR[0] + roiR[2], roiR[1] + roiR[3]), 255, -1)

    # 6. 应用掩码（两种方式可选）
    # 方式一：直接应用原始图像（不进行几何校正）
    maskedL = cv2.bitwise_and(imgL, imgL, mask=maskL)
    maskedR = cv2.bitwise_and(imgR, imgR, mask=maskR)

    # 方式二：先校正图像再应用掩码（更精确但会改变几何关系）
    # rectifiedL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    # rectifiedR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    # maskedL = cv2.bitwise_and(rectifiedL, rectifiedL, mask=maskL)
    # maskedR = cv2.bitwise_and(rectifiedR, rectifiedR, mask=maskR)

    return maskedL, maskedR, maskL, maskR


def visualize_results(img, masked_img, mask, win_name):
    """可视化结果（原始图/掩码图/掩码叠加）"""
    overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow(f"{win_name}_Original", img)
    cv2.imshow(f"{win_name}_Mask", mask)
    cv2.imshow(f"{win_name}_Overlay", overlay)
    cv2.imshow(f"{win_name}_Masked", masked_img)


if __name__ == "__main__":
    # 示例使用流程
    imgL = cv2.imread("/media/junf/C6769873769865C9/RedTilmpa/0509_0/testL/Image_229.png")
    imgR = cv2.imread("/media/junf/C6769873769865C9/RedTilmpa/0509_0/testR/Image_229.png")
    calib_file = "/media/junf/新加卷/开心摸鱼项目/MoChaOutputs/uw_photo02.yaml"

    # 生成掩码图像
    maskedL, maskedR, maskL, maskR = create_stereo_masks(imgL, imgR, calib_file)

    # 可视化结果
    # visualize_results(imgL, maskedL, maskL, "Left")
    # visualize_results(imgR, maskedR, maskR, "Right")

    # 保存结果
    cv2.imwrite("masked_left.jpg", maskedL)
    cv2.imwrite("masked_right.jpg", maskedR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()