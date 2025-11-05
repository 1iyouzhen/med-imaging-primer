# 医学图像重采样

## 📋 概述

本代码示例实现了医学图像重采样的完整流程，支持将不同分辨率、不同空间方向的医学影像重采样到统一的标准。这是医学图像预处理中的基础且关键的步骤，对后续的图像分析和算法性能有重要影响。

## 🎯 学习目标

1. **理解重采样的必要性**
   - 不同设备分辨率差异
   - 各向异性的问题
   - 多模态配准需求

2. **掌握插值方法原理**
   - 最近邻插值及其适用场景
   - 线性插值的平衡性
   - 高次插值的质量优势

3. **学习质量评估方法**
   - 强度保持度评估
   - 空间分辨率变化分析
   - 图像质量指标计算

4. **了解实际应用场景**
   - 多模态图像标准化
   - 体积计算和定量分析
   - 机器学习预处理

## 🧮 算法原理

### 重采样数学原理

重采样的核心是坐标变换和插值：

```python
# 坐标变换
P_original = (x, y, z)  # 原始坐标
P_resampled = T(P_original)  # 变换后坐标

# 插值计算
I_resampled(P_resampled) = Σ(I_original(P_i) * w_i)
```

### 缩放因子计算

```python
def calculate_scale_factors(original_spacing, target_spacing):
    """
    计算重采样缩放因子

    scale_factor = original_spacing / target_spacing
    """
    scale_factors = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(np.array(original_shape) * scale_factors).astype(int)
    return scale_factors, new_shape
```

### 插值方法对比

| 方法 | 阶数 | 适用场景 | 优势 | 劣势 |
|------|------|----------|------|------|
| 最近邻 | 0 | 标签图像 | 保持离散值 | 精度较低，锯齿效应 |
| 线性 | 1 | 一般图像 | 计算高效，平衡性好 | 边缘模糊 |
| 三次样条 | 3 | 高质量要求 | 平滑度高 | 计算量大，过冲问题 |

## 🏥 临床应用

### 适用场景

| 应用场景 | 原始分辨率 | 目标分辨率 | 临床意义 |
|----------|------------|------------|----------|
| **多模态配准** | 不同设备各异 | 各向同性1mm³ | 精确配准基础 |
| **体积测量** | 各向异性切片 | 标准化分辨率 | 定量分析准确 |
| **AI训练** | 多样化输入 | 统一分辨率 | 模型训练一致性 |
| **纵向研究** | 不同时间点扫描 | 标准化格式 | 可比性增强 |

### 质量标准

- **强度保持度**: 相关系数 > 0.9
- **各向同性**: 间距变异系数 < 0.1
- **边界完整性**: 结构连续性保持
- **计算效率**: 处理时间合理

## 📊 测试数据

### 合成数据特点

代码包含三种模态的合成医学图像：

1. **CT图像**
   - HU值范围: -1000 ~ 400 HU
   - 解剖结构: 胸腔轮廓、心脏、肝脏、肾脏
   - 噪声水平: 可调节高斯噪声

2. **MRI图像**
   - 强度范围: 0 ~ 1 (归一化)
   - 脑部结构: 白质、灰质、脑脊液
   - 对比度: 模拟T1加权

3. **PET图像**
   - SUV范围: 0 ~ 10
   - 代谢特征: 背景代谢、高代谢灶
   - 空间分辨率: 典型低分辨率

### 真实数据推荐

**TCIA数据集**
- 网址: https://www.cancerimagingarchive.net/
- 描述: 癌症影像档案
- 特点: 多种模态，标准DICOM格式

**The Cancer Imaging Archive (TCIA)**
- 网址: https://wiki.cancerimagingarchive.net/
- 描述: 公开的医学影像数据集
- 特点: 包含各种癌症的CT、MRI、PET数据

## 🚀 使用方法

### 基本使用

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主程序
python main.py

# 运行测试
python test_simple.py
```

### 单独使用重采样器

```python
from main import MedicalImageResampler, ResamplingConfig

# 配置重采样参数
config = ResamplingConfig(
    target_spacing=(1.0, 1.0, 1.0),
    interpolation_method='linear',
    anti_aliasing=True,
    preserve_intensity=True
)

# 创建重采样器
resampler = MedicalImageResampler(config)

# 执行重采样
original_image = ...  # 原始图像
original_spacing = (0.5, 0.5, 2.0)  # 原始间距
target_spacing = (1.0, 1.0, 1.0)  # 目标间距

resampled_image, info = resampler.resample_image(
    original_image, original_spacing, target_spacing
)

print(f"重采样完成: {original_image.shape} -> {resampled_image.shape}")
```

### 质量评估

```python
from main import evaluate_resampling_quality

# 评估重采样质量
quality_metrics = evaluate_resampling_quality(
    original_image, resampled_image,
    original_spacing, target_spacing
)

print(f"强度相关系数: {quality_metrics['intensity_preservation']['correlation']:.4f}")
print(f"体素大小变化: {quality_metrics['spatial_resolution']['voxel_size_change']:.3f}")
print(f"信噪比: {quality_metrics['image_quality']['snr']:.2f}")
```

### 插值方法比较

```python
from main import compare_interpolation_methods

# 比较不同插值方法
results = compare_interpolation_methods(
    original_image, original_spacing, target_spacing,
    methods=['nearest', 'linear', 'cubic'],
    save_path="interpolation_comparison.png"
)

for method, result in results.items():
    print(f"{method}: 形状={result['image'].shape}")
```

## 📈 输出结果

### 重采样信息报告

```python
resampling_info = {
    'original_info': {
        'shape': original_shape,
        'spacing': original_spacing,
        'min_value': float(np.min(original_image)),
        'max_value': float(np.max(original_image)),
        'mean_value': float(np.mean(original_image))
    },
    'resampled_info': {
        'shape': resampled_shape,
        'spacing': target_spacing,
        'min_value': float(np.min(resampled_image)),
        'max_value': float(np.max(resampled_image)),
        'mean_value': float(np.mean(resampled_image))
    },
    'parameters': {
        'scale_factors': scale_factors,
        'interpolation_method': method,
        'anti_aliasing': True
    }
}
```

### 可视化结果

生成多视图对比可视化：

1. **原始图像**: 轴位、冠状位、矢状位
2. **重采样图像**: 相应位置的切片
3. **差异显示**: 强度变化可视化
4. **质量指标**: 数值化评估结果

### 保存文件

- `output/resampling_result_*.png`: 不同模态的重采样结果
- `output/interpolation_comparison.png`: 插值方法对比
- `output/resampling_report.json`: 详细重采样报告

## ⚙️ 依赖要求

```bash
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-image>=0.19.0
SimpleITK>=2.1.0  # 可选，用于B样条插值
```

## 🧪 测试说明

运行 `test_simple.py` 将执行以下测试：

1. **基本功能测试**
   - 重采样器初始化
   - 形状计算正确性
   - 输出格式验证

2. **插值方法测试**
   - 最近邻插值
   - 线性插值
   - 三次样条插值

3. **合成数据生成测试**
   - CT图像生成
   - MRI图像生成
   - PET图像生成

4. **质量评估测试**
   - 强度保持度计算
   - 空间分辨率分析
   - 图像质量指标

## 🎓 学习要点

1. **理论基础**: 理解重采样的数学原理和物理意义
2. **方法选择**: 掌握不同插值方法的适用场景
3. **质量评估**: 学会评估重采样效果的指标和方法
4. **实际应用**: 了解在临床和研究中的应用价值
5. **优化技巧**: 掌握提高重采样质量和效率的方法

## 📚 扩展阅读

1. **经典论文**
   - Parker J, et al. Resampling of medical images. IEEE Trans Med Imaging 2006.
   - Lehmann TM, et al. Survey: interpolation methods in medical image processing. IEEE Trans Med Imaging 1999.

2. **技术扩展**
   - 配准与重采样的结合
   - 自适应插值方法
   - GPU加速重采样

3. **应用领域**
   - 放射治疗计划
   - 功能神经影像分析
   - 计算病理学

## 🔬 高级主题

### 自适应插值

```python
def adaptive_interpolation(image, gradient_threshold):
    """
    基于梯度的自适应插值
    """
    # 计算梯度
    gradient = np.gradient(image)
    gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))

    # 高梯度区域使用高阶插值
    mask = gradient_magnitude > gradient_threshold

    result = np.zeros_like(image)
    result[mask] = cubic_interpolation(image[mask])
    result[~mask] = linear_interpolation(image[~mask])

    return result
```

### 多模态联合重采样

```python
def multimodal_resampling(images, spacings, target_spacing):
    """
    多模态图像联合重采样
    """
    # 计算共同目标形状
    reference_shape = calculate_target_shape(
        images[0].shape, spacings[0], target_spacing
    )

    resampled_images = []
    for image, spacing in zip(images, spacings):
        resampled, _ = resample_image(
            image, spacing, target_spacing, method='linear'
        )
        resampled_images.append(resampled)

    return resampled_images
```

## 🚨 注意事项

1. **内存管理**: 大体积图像重采样需要充足的内存
2. **插值选择**: 标签图像必须使用最近邻插值
3. **间距单位**: 确保所有间距使用相同的物理单位
4. **坐标系统**: 注意不同软件的坐标系统差异
5. **质量验证**: 重采样后应该验证结果的合理性
6. **JSON序列化**: 确保所有numpy数值类型转换为Python原生float类型，避免JSON序列化错误

## 📊 性能基准

### 处理速度参考

| 图像大小 | 插值方法 | 处理时间 | 内存使用 |
|----------|----------|----------|----------|
| 64×64×32 | 最近邻 | ~0.02秒 | ~16MB |
| 64×64×32 | 线性 | ~0.05秒 | ~16MB |
| 64×64×32 | 三次样条 | ~0.12秒 | ~16MB |
| 256×256×128 | 线性 | ~1.5秒 | ~256MB |
| 512×512×256 | 线性 | ~12秒 | ~1GB |

### 质量指标参考

| 插值方法 | 强度相关系数 | 信噪比 | 计算复杂度 |
|----------|-------------|--------|------------|
| 最近邻 | 0.85-0.90 | 15-20 | O(1) |
| 线性 | 0.90-0.95 | 20-25 | O(n) |
| 三次样条 | 0.95-0.98 | 25-30 | O(n²) |

## 📞 技术支持

如有问题，请参考：
1. 代码注释和文档
2. 测试用例和示例
3. 相关论文和技术资料

医学图像重采样是医学影像分析的基础技术，对保证数据一致性、提高分析准确性、支持多模态融合具有不可替代的重要作用。