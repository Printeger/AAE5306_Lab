# AAE5306 Lab: Stereo Vision Processing with EuRoC MAV Dataset

## 概述 (Overview)

本实验使用EuRoC MAV无人机数据集，实现完整的立体视觉处理流水线，包括相机标定、图像提取、特征检测、特征匹配和深度恢复。

This laboratory implements a complete stereo vision processing pipeline using the EuRoC MAV drone dataset, including camera calibration extraction, image extraction, feature detection, feature matching, and depth recovery.

## 项目结构 (Project Structure)

```
AAE5306_Lab/
├── config.yaml                  # 主配置文件 (Main configuration file)
├── utils.py                     # 工具函数 (Utility functions)
├── 1. extract_calibration.py   # 任务1: 相机标定提取
├── 2. extract_images.py        # 任务2: 图像提取
├── 3. feature_detection.py     # 任务3: 特征检测
├── 4. feature_matching.py      # 任务4: 特征匹配
├── 5. depth_recovery.py        # 任务5: 深度恢复
├── data/                        # 数据目录
│   ├── euroc/                  # EuRoC数据集
│   │   ├── MH_01_easy.yaml    # 标定文件
│   │   └── rosbags/
│   │       └── MH_01_easy.bag # ROS bag文件
│   └── extracted_images/       # 提取的图像
│       ├── cam0/               # 左相机图像
│       ├── cam1/               # 右相机图像
│       └── timestamps.txt      # 时间戳
├── config/                      # 配置输出目录
│   └── euroc_calibration.yaml  # 处理后的标定
├── results/                     # 结果目录
│   ├── features/               # 特征检测结果
│   ├── matches/                # 特征匹配结果
│   └── depth/                  # 深度恢复结果
└── README.md                    # 本文件
```

## 环境要求 (Requirements)

### Python版本
- Python 3.7+

### 依赖包 (Dependencies)
```bash
# 核心依赖
pip install numpy
pip install opencv-contrib-python  # 包含SIFT
pip install matplotlib
pip install pyyaml

# ROS相关 (仅用于任务2)
pip install rospkg
# 或使用系统的ROS安装: sudo apt-get install ros-noetic-cv-bridge
```

### 系统要求 (System Requirements)
- 如果需要运行任务2(图像提取)，需要安装ROS Noetic
- Windows用户可以跳过任务2，直接使用预提取的图像

## 配置说明 (Configuration)

所有参数均在 `config.yaml` 文件中配置。主要配置项包括：

### 1. 路径配置 (Paths Configuration)
```yaml
paths:
  euroc_calibration: "data/euroc/MH_01_easy.yaml"
  rosbag_file: "data/euroc/rosbags/MH_01_easy.bag"
  output_base: "results"
  # ... 其他路径
```

### 2. 任务开关 (Task Switches)
每个任务都有 `enabled` 选项来控制是否执行：
```yaml
calibration:
  enabled: true

image_extraction:
  enabled: true
  
# ... 其他任务
```

### 3. 图像提取参数 (Image Extraction Parameters)
```yaml
image_extraction:
  skip_frames: 10      # 每10帧提取一次
  max_pairs: 300       # 最多提取300对图像
  cam0_topic: "/cam0/image_raw"
  cam1_topic: "/cam1/image_raw"
```

### 4. 特征检测参数 (Feature Detection Parameters)
```yaml
feature_detection:
  detectors:           # 使用的检测器
    - sift
    - orb
    - fast
    - harris
  test_image_index: 50 # 使用第50对图像进行测试
  visualize: true      # 生成可视化
```

### 5. 特征匹配参数 (Feature Matching Parameters)
```yaml
feature_matching:
  detectors:
    - sift
    - orb
  sift:
    ratio_threshold: 0.75      # Lowe比率测试阈值
    epipolar_threshold: 2.0    # 极线约束阈值(像素)
  orb:
    ratio_threshold: 0.75
    epipolar_threshold: 2.0
```

### 6. 深度恢复参数 (Depth Recovery Parameters)
```yaml
depth_recovery:
  detector: sift       # 使用哪个检测器的匹配结果
  min_depth: 0.5      # 最小有效深度(米)
  max_depth: 15.0     # 最大有效深度(米)
  visualize: true     # 生成可视化
```

## 使用方法 (Usage)

### 准备工作 (Preparation)

1. **下载EuRoC数据集**
```bash
# 创建数据目录
mkdir -p data/euroc/rosbags

# 下载标定文件
cd data/euroc
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.yaml

# 下载ROS bag文件
cd rosbags
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.bag
```

2. **配置参数**
编辑 `config.yaml` 文件，根据需要调整参数。

### 运行步骤 (Execution Steps)

#### 任务1: 提取相机标定 (Task 1: Extract Camera Calibration)

**输入 (Input):**
- `data/euroc/MH_01_easy.yaml` - EuRoC标定文件

**输出 (Output):**
- `config/euroc_calibration.yaml` - OpenCV兼容的标定文件
- 终端显示标定参数摘要

**运行 (Run):**
```bash
python "1. extract_calibration.py"
```

**配置参数 (Configuration):**
- 无需额外配置，只需确保输入文件路径正确

---

#### 任务2: 提取立体图像对 (Task 2: Extract Stereo Image Pairs)

**输入 (Input):**
- `data/euroc/rosbags/MH_01_easy.bag` - ROS bag文件

**输出 (Output):**
- `data/extracted_images/cam0/` - 左相机图像 (000000.png, 000001.png, ...)
- `data/extracted_images/cam1/` - 右相机图像
- `data/extracted_images/timestamps.txt` - 时间戳文件

**运行 (Run):**
```bash
python "2. extract_images.py"
```

**配置参数 (Configuration in config.yaml):**
```yaml
image_extraction:
  skip_frames: 10      # 调整采样频率
  max_pairs: 300       # 调整提取数量
  cam0_topic: "/cam0/image_raw"
  cam1_topic: "/cam1/image_raw"
  info_only: false     # 设为true仅查看bag信息
```

**注意 (Note):** Windows用户如果无法安装ROS，可以：
- 使用Linux虚拟机运行此任务
- 或从教师处获取预提取的图像

---

#### 任务3: 特征检测 (Task 3: Feature Detection)

**输入 (Input):**
- 提取的图像对 (来自任务2)
- 通过 `test_image_index` 指定使用哪对图像

**输出 (Output):**
- `results/features/cam0/` - 左图像特征可视化
  - `sift_features.png`
  - `orb_features.png`
  - `fast_features.png`
  - `harris_features.png`
- `results/features/cam1/` - 右图像特征可视化
- `results/features/cam0_comparison.png` - 性能对比图
- `results/features/cam1_comparison.png`
- `results/features/detection_statistics.json` - 统计数据
- `results/features/comparison_summary.txt` - 文本摘要

**运行 (Run):**
```bash
python "3. feature_detection.py"
```

**配置参数 (Configuration in config.yaml):**
```yaml
feature_detection:
  detectors:           # 选择要使用的检测器
    - sift
    - orb
    - fast
    - harris
  test_image_index: 50 # 选择测试图像 (0-299)
  visualize: true      # 是否生成可视化
  save_statistics: true
```

**输出说明:**
- **num_features**: 检测到的特征点数量
- **computation_time_ms**: 计算时间(毫秒)
- **has_descriptors**: 是否包含描述子(FAST和Harris不包含)

---

#### 任务4: 特征匹配 (Task 4: Feature Matching)

**输入 (Input):**
- 提取的图像对 (来自任务2)
- 使用与任务3相同的 `test_image_index`

**输出 (Output):**
- `results/matches/sift_matches.pkl` - SIFT匹配数据(用于任务5)
- `results/matches/orb_matches.pkl` - ORB匹配数据
- `results/matches/sift_matching_stats.json` - 匹配统计
- `results/matches/orb_matching_stats.json`
- `results/matches/sift_matches.png` - 匹配可视化
- `results/matches/orb_matches.png`

**运行 (Run):**
```bash
python "4. feature_matching.py"
```

**配置参数 (Configuration in config.yaml):**
```yaml
feature_matching:
  detectors:
    - sift             # 只有SIFT和ORB支持匹配
    - orb
  test_image_index: 50 # 应与任务3相同
  
  sift:
    ratio_threshold: 0.75      # Lowe比率测试 (0.7-0.8推荐)
    epipolar_threshold: 2.0    # 极线约束(像素)
  
  orb:
    ratio_threshold: 0.75
    epipolar_threshold: 2.0
  
  visualize: true
  max_display_matches: 100     # 可视化中显示的最大匹配数
```

**参数调优建议:**
- **ratio_threshold**: 降低值(如0.7)得到更可靠但更少的匹配
- **epipolar_threshold**: 对于已校正的立体图像，2.0像素是合理值
- 调整参数后重新运行以观察匹配质量变化

**输出说明:**
- **num_initial_matches**: 初始匹配数
- **num_ratio_filtered**: 比率测试后的匹配数
- **num_final_matches**: 极线约束后的最终匹配数
- **disparity_mean/std**: 视差统计

---

#### 任务5: 深度恢复 (Task 5: Depth Recovery)

**输入 (Input):**
- `results/matches/{detector}_matches.pkl` - 匹配数据(来自任务4)
- `config/euroc_calibration.yaml` - 标定数据(来自任务1)
- 左图像(用于点云着色)

**输出 (Output):**
- `results/depth/point_cloud_{detector}.ply` - 3D点云(可用CloudCompare/MeshLab查看)
- `results/depth/depth_statistics_{detector}.json` - 深度统计
- `results/depth/depth_map_{detector}.png` - 深度图可视化
- `results/depth/pointcloud_2d_{detector}.png` - 点云2D投影

**运行 (Run):**
```bash
python "5. depth_recovery.py"
```

**配置参数 (Configuration in config.yaml):**
```yaml
depth_recovery:
  detector: sift       # 使用哪个检测器的匹配结果
  min_depth: 0.5      # 过滤小于此值的深度(米)
  max_depth: 15.0     # 过滤大于此值的深度(米)
  
  use_depth_coloring: true  # 使用深度着色点云
  
  visualize: true
  generate_depth_map: true
  generate_2d_projections: true
```

**参数调优建议:**
- 根据场景调整 **min_depth** 和 **max_depth**
- 室内场景: 0.5-10米
- 室外场景: 1-50米
- 调整范围可以过滤噪声点

**输出说明:**
- **num_valid_depth**: 有效3D点数量
- **mean_depth**: 平均深度(米)
- **depth_range**: 深度范围[最小, 最大]
- **PLY文件**: 可用以下软件打开
  - CloudCompare (推荐)
  - MeshLab
  - Open3D

---

### 完整流水线运行 (Run Complete Pipeline)

按顺序运行所有任务：

```bash
# 任务1: 提取标定
python "1. extract_calibration.py"

# 任务2: 提取图像 (需要ROS)
python "2. extract_images.py"

# 任务3: 特征检测
python "3. feature_detection.py"

# 任务4: 特征匹配
python "4. feature_matching.py"

# 任务5: 深度恢复
python "5. depth_recovery.py"
```

## 配置文件详解 (Configuration File Details)

### 输入输出总结 (Input/Output Summary)

| 任务 | 输入 | 输出 | 可配置参数 |
|------|------|------|-----------|
| 1. 标定提取 | `MH_01_easy.yaml` | `euroc_calibration.yaml` | 无 |
| 2. 图像提取 | `MH_01_easy.bag` | `extracted_images/` | `skip_frames`, `max_pairs`, `topics` |
| 3. 特征检测 | 图像对 | 可视化, 统计 | `detectors`, `test_image_index` |
| 4. 特征匹配 | 图像对 | 匹配数据, 可视化 | `ratio_threshold`, `epipolar_threshold` |
| 5. 深度恢复 | 匹配数据, 标定 | 点云, 深度图 | `min_depth`, `max_depth`, `detector` |

### 关键参数说明 (Key Parameters)

#### test_image_index
- **范围**: 0 到 (max_pairs - 1)
- **默认**: 50
- **说明**: 选择哪对图像进行特征检测、匹配和深度恢复
- **建议**: 
  - 选择中间帧(如50)避免数据集开始/结束的边界效应
  - 选择纹理丰富的场景以获得更多特征

#### ratio_threshold
- **范围**: 0.6 - 0.9
- **默认**: 0.75
- **说明**: Lowe's ratio test阈值，用于过滤模糊匹配
- **建议**:
  - 0.7: 高精度，低召回率
  - 0.8: 平衡
  - 0.9: 低精度，高召回率

#### epipolar_threshold
- **范围**: 1.0 - 5.0 (像素)
- **默认**: 2.0
- **说明**: 立体图像对的y坐标最大差值
- **建议**:
  - 已校正图像: 1.0-2.0
  - 未校正图像: 3.0-5.0

#### min_depth / max_depth
- **范围**: > 0 (米)
- **默认**: 0.5 / 15.0
- **说明**: 有效深度范围，用于过滤异常值
- **建议**:
  - 根据具体场景调整
  - 过小的min_depth会包含近处噪声
  - 过大的max_depth会包含远处低精度点

## 常见问题 (FAQ)

### Q1: 如何更改处理的图像对？
**A:** 修改 `config.yaml` 中的 `test_image_index` 参数，然后重新运行任务3-5。

### Q2: 为什么FAST和Harris没有匹配结果？
**A:** 这两个检测器只检测关键点，不生成描述子，因此无法用于匹配。只有SIFT和ORB支持特征匹配。

### Q3: 如何提高匹配数量？
**A:** 
1. 增加 `ratio_threshold` (如从0.75到0.8)
2. 增加 `epipolar_threshold` (如从2.0到3.0)
3. 选择纹理更丰富的图像对

### Q4: 点云文件如何查看？
**A:** 使用以下任一软件：
- CloudCompare (免费，推荐)
- MeshLab (免费)
- Open3D (Python库)

### Q5: Windows系统无法运行任务2怎么办？
**A:** 
1. 使用Linux虚拟机(Ubuntu 20.04 + ROS Noetic)
2. 从教师处获取预提取的图像
3. 使用WSL2 + ROS

### Q6: 如何批量处理多对图像？
**A:** 可以编写简单的脚本循环修改 `test_image_index` 并运行：
```python
import yaml
import subprocess

config_file = 'config.yaml'
for idx in range(0, 100, 10):  # 每10帧处理一次
    # 加载配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改索引
    config['feature_detection']['test_image_index'] = idx
    config['feature_matching']['test_image_index'] = idx
    
    # 保存配置
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # 运行任务
    subprocess.run(['python', '3. feature_detection.py'])
    subprocess.run(['python', '4. feature_matching.py'])
    subprocess.run(['python', '5. depth_recovery.py'])
```

## 实验报告建议 (Lab Report Suggestions)

您的报告应包括：

1. **方法描述**
   - 简述每个任务的原理
   - 说明使用的算法(SIFT, ORB等)

2. **参数选择**
   - 列出使用的主要参数值
   - 解释为什么选择这些值

3. **结果展示**
   - 特征检测对比图
   - 特征匹配可视化
   - 深度图和点云
   - 统计数据表格

4. **分析讨论**
   - 比较不同特征检测器的性能
   - 分析匹配质量
   - 讨论深度恢复的精度
   - 参数调优的影响

5. **结论**
   - 总结实验发现
   - 提出改进建议

## 参考资料 (References)

1. **EuRoC MAV Dataset**
   - Paper: Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 2016
   - Website: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2. **Feature Detection**
   - SIFT: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004
   - ORB: Rublee et al., "ORB: An efficient alternative to SIFT or SURF", ICCV 2011

3. **Stereo Vision**
   - Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2nd Edition
   - OpenCV Stereo Documentation: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

## 致谢 (Acknowledgments)

本实验基于ETH Zurich的EuRoC MAV数据集开发。感谢开源社区提供的OpenCV、ROS等优秀工具。

## 许可证 (License)

本代码仅供教学使用。

---

**最后更新 (Last Updated):** 2025-10-29  
**作者 (Author):** AAE5306 Teaching Team  
**课程 (Course):** AAE5306 - 低空经济中的视觉技术
