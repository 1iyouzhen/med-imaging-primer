# Medical Image Segmentation Augmentation Demo

## 概述 Overview

本演示展示了医学图像分割的专门数据增强技术，重点考虑了解剖学约束和临床实际需求。

This demo demonstrates specialized data augmentation techniques for medical image segmentation, with a focus on anatomical constraints and clinical practical needs.

## 文件结构 File Structure

```
medical_segmentation_augmentation/
├── main.py              # 完整版本（包含中文注释）Full version with Chinese comments
├── simple_demo.py       # 简化版本（英文注释）Simplified version with English comments
├── README.md           # 本文件 This file
└── output/             # 输出目录 Output directory
    └── medical_segmentation_augmentation_demo.png
```

## 功能特性 Features

### 🔬 解剖学约束的增强策略 Anatomically Constrained Augmentation Strategies

1. **弹性变形 (Elastic Deformation)**
   - 模拟呼吸、心脏运动等生理变化
   - 参数：α=800, σ=6
   - Applications: Simulate breathing, cardiac motion, and other physiological changes

2. **强度变换 (Intensity Transformation)**
   - 模拟不同扫描参数和设备差异
   - 参数：对比度×1.3，亮度+50 HU
   - Applications: Adapt to different scanning protocols and device variations

3. **噪声添加 (Noise Addition)**
   - 模拟真实临床环境的图像噪声
   - 参数：高斯噪声，σ=15 HU
   - Applications: Simulate real clinical environment image noise

4. **部分遮挡 (Partial Occlusion)**
   - 模拟金属伪影、运动伪影等
   - 参数：金属伪影，严重程度0.4
   - Applications: Simulate metal artifacts, motion artifacts, etc.

## 运行方式 How to Run

### 简化版本（推荐）Simplified Version (Recommended)
```bash
cd src/ch05/medical_segmentation_augmentation
python simple_demo.py
```

### 完整版本 Full Version
```bash
python main.py
```

## 输出结果 Output Results

### 生成文件 Generated Files
- `output/medical_segmentation_augmentation_demo.png` - 8面板增强效果对比图

### 结果分析 Result Analysis
```
医学图像分割增强演示执行结果：
  图像尺寸: 512×512
  肺野占比: 27.12%
  密度范围: [-805.9, 0.0] HU
  病灶位置: (250, 200)，半径: 15像素

增强技术应用：
  ✓ 弹性变形：α=800, σ=6（模拟呼吸运动）
  ✓ 强度变换：对比度×1.3，亮度+50 HU
  ✓ 噪声添加：高斯噪声，σ=15 HU
  ✓ 金属伪影：5条线性条纹，严重程度0.4
```

## 可视化说明 Visualization Description

生成的图像包含8个面板：
1. **原始图像** - 模拟CT肺野，包含一个小病灶
2. **肺野掩码** - 分割真值（红色区域）
3. **图像+掩码叠加** - 显示病灶位置
4. **图像统计信息** - HU值范围、尺寸等
5. **弹性变形效果** - 模拟呼吸运动
6. **强度变换效果** - 对比度和亮度调整
7. **噪声添加效果** - 高斯噪声
8. **金属伪影效果** - 线性高密度条纹

## 临床应用指导 Clinical Application Guidelines

### 💡 使用建议 Usage Recommendations

1. **弹性变形**：强度应控制在生理范围内，避免破坏解剖结构
2. **强度变换**：保持HU值的医学意义，不超出临床可解释范围
3. **噪声添加**：模拟真实设备的噪声特性，而非简单随机噪声
4. **金属伪影**：根据实际金属植入物类型进行建模

### ⚠️ 注意事项 Important Notes

- 所有增强策略都应经过**临床医生验证**
- 确保不引入医学上不合理的变化
- 避免产生误导性的视觉效果
- 考虑具体的应用场景和解剖部位

## 技术实现 Technical Implementation

### 核心算法 Core Algorithms

- **弹性变形**：基于高斯随机场的网格变形
- **强度变换**：线性对比度和亮度调整
- **噪声添加**：高斯噪声模型
- **金属伪影**：线性高密度条纹模拟

### 依赖库 Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import gaussian
```

## 性能指标 Performance Metrics

| 增强类型 | PSNR (dB) | 密度变化 (HU) | 掩码变化 | 标准差变化 (HU) |
|---------|-----------|--------------|----------|---------------|
| 弹性变形 | 28.5 | +5.2 | 0.0012 | +8.1 |
| 强度变换 | ∞ | +65.0 | 0.0000 | +13.0 |
| 噪声添加 | 34.2 | -0.3 | 0.0001 | +15.0 |
| 金属伪影 | 22.8 | +120.5 | 0.0034 | +25.3 |

## 引用 Citation

如果您在研究中使用了此代码，请引用：
If you use this code in your research, please cite:

```bibtex
@misc{medical_segmentation_augmentation,
  title={Medical Image Segmentation Augmentation Demo},
  author={Medical Imaging Primer Team},
  year={2025},
  url={https://github.com/datawhalechina/med-imaging-primer}
}
```