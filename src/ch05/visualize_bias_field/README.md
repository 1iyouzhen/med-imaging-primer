# MRI偏场场可视化工具

## 📋 概述

本代码示例实现了MRI图像中偏场场的检测、可视化和校正效果分析。偏场场是MRI图像中常见的强度不均匀现象，主要由RF线圈敏感度不均匀、梯度场非线性、组织磁化率差异等因素引起。

## 🎯 学习目标

1. **理解MRI偏场场的形成机理**
   - RF线圈敏感度不均匀的影响
   - 梯度场非线性效应
   - 组织磁化率差异
   - 患者体型和位置因素

2. **掌握偏场场的可视化方法**
   - 差分法：I_observed / I_corrected
   - 对数域：log(I_observed) - log(I_corrected)
   - 滤波法：低通滤波提取偏场场

3. **了解偏场校正算法的效果评估**
   - 定量评估指标（CV、PSNR、SSIM）
   - 可视化分析方法
   - 不同校正方法的比较

## 🧮 算法原理

### 偏场场数学模型

MRI图像观测模型可以表示为：

```
I_observed(x) = B(x) × I_true(x) + noise
```

其中：
- `I_observed(x)`: 观测到的MRI信号强度
- `B(x)`: 偏场场（空间变化的增益场）
- `I_true(x)`: 真实的组织信号强度
- `noise`: 噪声

### 可视化方法

1. **除法方法**
   ```python
   bias_field = original_image / corrected_image
   ```

2. **对数差分方法**
   ```python
   bias_field_log = log(original_image) - log(corrected_image)
   bias_field = exp(bias_field_log)
   ```

3. **滤波方法**
   ```python
   bias_field = gaussian_filter(original_image, sigma=20)
   bias_field = bias_field / mean(bias_field)
   ```

## 🚀 使用方法

### 基本使用

```bash
# 运行主程序
python main.py
```

### 主要功能

1. **合成数据生成**
   - 真实的脑部解剖结构模拟
   - 多尺度偏场场生成
   - 噪声添加和强度调整

2. **偏场场校正模拟**
   - 高斯滤波方法
   - 同态滤波方法
   - 多项式拟合方法

3. **多种可视化方法**
   - 原始图像、偏场场、校正图像对比
   - 强度分布直方图分析
   - 剖面线对比分析

## 📈 输出结果

### 生成的可视化文件

运行代码会在 `output/` 文件夹中生成以下文件：

1. **偏场场可视化结果**
   - `bias_field_visualization_division.png`: 除法方法结果
   - `bias_field_visualization_log_diff.png`: 对数差分方法结果
   - `bias_field_visualization_filter.png`: 滤波方法结果

2. **校正方法对比**
   - `bias_field_gaussian_*.png`: 高斯校正方法结果
   - `bias_field_homomorphic_*.png`: 同态校正方法结果
   - `bias_field_polynomial_*.png`: 多项式校正方法结果

3. **综合对比图**
   - `bias_field_methods_comparison.png`: 所有方法的综合对比

### 量化评估指标

代码会输出详细的统计信息：

```
偏场场分析统计:
原始图像 - 均值: 0.24, 标准差: 0.31, 变异系数: 1.278
校正图像 - 均值: 0.19, 标准差: 0.19, 变异系数: 0.972
偏场场 - 均值: 0.933, 标准差: 1.058, 范围: [0.000, 3.297]
校正效果 - CV减少: 24.0%, 相关系数: 0.468
```

### 方法对比结果

不同校正方法的性能对比：

```
Method 1 - MSE: 0.0958, PSNR: 10.2, SSIM: 0.368  (高斯方法)
Method 2 - MSE: 0.1984, PSNR: 7.0, SSIM: 0.149   (同态方法)
Method 3 - MSE: 0.0663, PSNR: 11.8, SSIM: 0.545   (多项式方法)
```

## 🧪 测试结果分析

### 偏场场检测效果

- ✅ **除法方法**: 简单直观，适用于轻微偏场
- ✅ **对数差分方法**: 增强小差异，适用于低对比度区域
- ✅ **滤波方法**: 平滑稳定，适用于噪声较大的图像

### 校正算法性能

1. **高斯滤波方法**
   - 优点：计算快速，稳定可靠
   - 缺点：可能过度平滑细节
   - 适用：一般偏场场校正

2. **同态滤波方法**
   - 优点：保持动态范围
   - 缺点：对噪声敏感
   - 适用：高对比度图像

3. **多项式拟合方法**
   - 优点：精确建模复杂偏场
   - 缺点：计算复杂，可能过拟合
   - 适用：严重偏场场

## 🎓 学习要点

1. **物理理解**: MRI偏场场的成因和影响因素
2. **数学建模**: 偏场场的数学表示和估计方法
3. **可视化技术**: 多种偏场场显示方法的优缺点
4. **算法评估**: 定量评估指标的计算和解释
5. **方法选择**: 根据数据特点选择合适的校正方法

## 📊 性能基准

基于合成数据的测试结果：

| 指标 | 高斯方法 | 同态方法 | 多项式方法 |
|------|----------|----------|------------|
| MSE | 0.0958 | 0.1984 | 0.0663 |
| PSNR | 10.2 dB | 7.0 dB | 11.8 dB |
| SSIM | 0.368 | 0.149 | 0.545 |
| 处理时间 | ~2秒 | ~3秒 | ~5秒 |

## ⚙️ 技术要求

```bash
pip install numpy matplotlib scipy scikit-image
```

## 🚨 注意事项

1. **数据范围**: 确保输入数据在合理范围内 [0, 1]
2. **数值稳定性**: 避免除零错误，添加小常数保护
3. **内存管理**: 大图像需要足够的内存空间
4. **参数调整**: 根据具体数据调整滤波参数

## 📚 扩展阅读

1. **经典论文**
   - Sled JG, et al. A nonparametric method for automatic correction of intensity nonuniformity in MRI data. IEEE TMI. 1998.
   - Tustison NJ, et al. N4ITK: improved N3 bias correction. IEEE TMI. 2010.

2. **相关算法**
   - N3 bias correction
   - N4ITK bias correction
   - Homomorphic filtering

本实现为教学目的的简化版本，展示了偏场场可视化的核心概念和方法。