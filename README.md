# 双目立体视觉图像处理工具

这是一个用于双目立体视觉图像处理的Python工具包，专门用于水下或复杂环境下的双目图像校正、增强和立体校正处理。

## 🌟 主要功能

- **彩色图像CLAHE增强**：在HSV色彩空间中对亮度通道应用CLAHE算法，保留原始颜色信息
- **图像去畸变**：使用相机标定参数消除镜头畸变
- **立体校正**：对左右图像进行立体校正，使极线水平对齐
- **图像拼接与极线显示**：生成带有极线的左右图像拼接结果
- **多进程加速**：支持多核并行处理，提高处理效率
- **批量处理**：支持文件夹内所有图像的批量处理

## 📁 项目结构

```
MoChaOutputs/
├── process_images.py          # 主要的图像处理脚本
├── process_images_batch.py    # 批量处理脚本
├── images_to_video.py         # 图像序列转视频
├── utils/                     # 工具模块
│   ├── undistortion.py       # 核心处理函数
│   └── exportSamePic.py      # 图像导出工具
├── mask.py                   # 掩膜处理工具
├── 时间戳匹配.py              # 时间戳同步工具
├── 批量转3通道.py             # 通道转换工具
├── *.yaml                    # 相机标定参数文件
└── README.md                 # 项目说明文档
```

## 🔧 环境要求

### 系统要求
- Python 3.7+
- Linux/Windows/macOS

### 依赖包
```bash
pip install opencv-python
pip install numpy
pip install pyyaml
pip install tqdm
pip install pathlib
```

或使用requirements.txt安装：
```bash
pip install -r requirements.txt
```

## 📖 使用方法

### 1. 准备工作

#### 1.1 图像准备
确保您有以下文件结构：
```
your_dataset/
├── 左眼(DA5182144)/     # 左相机图像文件夹
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── 右眼(DA5182138)/     # 右相机图像文件夹
    ├── image001.png
    ├── image002.png
    └── ...
```

#### 1.2 相机标定文件
准备YAML格式的相机标定文件，包含以下参数：
- `camera_matrix_left`: 左相机内参矩阵
- `dist_coeff_left`: 左相机畸变系数
- `camera_matrix_right`: 右相机内参矩阵
- `dist_coeff_right`: 右相机畸变系数
- `R`: 旋转矩阵
- `T`: 平移向量

### 2. 配置参数

编辑 `process_images.py` 中的配置参数：

```python
# 配置参数
left_folder = Path(r'/path/to/your/左眼(DA5182144)')
right_folder = Path(r'/path/to/your/右眼(DA5182138)')
output_base = Path(r'/path/to/your/output/')
calib_file = Path(r'/path/to/your/calibration.yaml')

# 图像尺寸
height = 1080
width = 1440
```

### 3. 运行处理

#### 3.1 批量处理
```bash
python process_images.py
```
可以使用代码中的multiprocessing部分内容，调整处理的核心数，使用机械硬盘的用户建议使用单进程处理
```python
    # 设置进程池大小
    num_processes = max(1, int(cpu_count() * 0.2))  # 拥有高速固态硬盘的用户可以使用该进程池设置甚至调整到更极端的参数

    num_processes = 1  # 机械硬盘用户
```


#### 3.2 生成视频
```bash
python images_to_video.py
```

## 📊 输出结果

处理完成后，将在输出目录下生成以下文件夹：

```
output_directory/
└── processing/
    ├── rectified_L/          # 校正后的左图像
    │   ├── image001.png
    │   └── ...
    ├── rectified_R/          # 校正后的右图像
    │   ├── image001.png
    │   └── ...
    └── Combined/             # 拼接图像（带极线）
        ├── image001.png
        └── ...
```

## ⚙️ 核心功能详解

### 彩色CLAHE增强
```python
from utils.undistortion import preprocess_color

# 对彩色图像应用CLAHE，保留颜色信息
enhanced1, enhanced2 = preprocess_color(img1, img2)
```

**技术原理**：
- 将BGR图像转换为HSV色彩空间
- 仅对V（亮度）通道应用CLAHE增强
- 保持H（色调）和S（饱和度）通道不变
- 转换回BGR格式输出

### 立体校正
```python
from utils.undistortion import getRectifyTransform, rectify_image

# 获取校正变换矩阵
map1x, map1y, map2x, map2y, Q, roi1, roi2 = getRectifyTransform(height, width, calib_file)

# 应用立体校正
rectified1, rectified2 = rectify_image(undist1, undist2, map1x, map1y, map2x, map2y)
```

## 🎛️ 参数调节

### CLAHE参数调节
在 `utils/undistortion.py` 中的 `preprocess_color` 函数：

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,      # 对比度限制，建议范围：1.0-4.0
    tileGridSize=(8, 8) # 网格大小，建议范围：(4,4)-(16,16)
)
```

### 多进程参数
在 `process_images.py` 中：

```python
# 进程数量（CPU核心数的20%）
num_processes = max(1, int(cpu_count() * 0.2))
```

## 🚀 性能优化

### 多进程处理
- 默认使用CPU核心数的20%进行并行处理
- 可根据系统性能调整进程数量
- 使用tqdm显示处理进度

### 内存管理
- 及时释放CLAHE对象：`cv2.CLAHE.collectGarbage(clahe)`
- 使用Path对象进行文件路径操作
- 支持大批量文件处理

## 📝 注意事项

1. **图像格式**：支持PNG、BMP等常见格式
2. **同名文件**：左右文件夹中必须有同名的图像文件
3. **内存使用**：处理大量高分辨率图像时注意内存使用情况
4. **标定质量**：确保相机标定参数的准确性，这直接影响处理效果
5. **颜色空间**：输入图像应为BGR格式（OpenCV默认格式）

## 🛠️ 常见问题

### Q: 处理后的图像出现黑边
A: 这是立体校正的正常现象，可以通过调整ROI参数来裁剪有效区域。

### Q: CLAHE增强效果不明显
A: 可以调整 `clipLimit` 参数（增大值）或 `tileGridSize`（减小值）。

### Q: 处理速度太慢
A: 可以增加进程数量或减少图像分辨率。

### Q: 极线不水平
A: 检查相机标定参数是否正确，特别是旋转矩阵R和平移向量T。

## 📞 技术支持

如有问题或建议，请通过以下方式联系：
- 创建Issue在项目仓库
- 查看代码注释获取技术细节
- 参考Jupyter Notebook中的开发示例

## 📄 许可证

本项目使用MIT许可证，详见LICENSE文件。

---

**开发环境**: Python 3.7+, OpenCV 4.5+, NumPy 1.21+ 