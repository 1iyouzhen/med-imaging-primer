#!/usr/bin/env python3
"""
MRI偏场场可视化工具
实现MRI图像中偏场场的检测、可视化和校正效果分析

学习目标：
1. 理解MRI偏场场的形成机理和特征
2. 掌握偏场场的可视化方法
3. 了解偏场校正算法的效果评估

算法原理：
MRI偏场场（Bias Field）是MRI图像中常见的强度不均匀现象，
主要来源于：
1. RF线圈敏感度不均匀
2. 梯度场非线性
3. 组织磁化率差异
4. 患者体型和位置

偏场场模型：
I_observed(x) = B(x) × I_true(x) + noise

其中：
- I_observed(x): 观测到的MRI信号强度
- B(x): 偏场场（空间变化的增益场）
- I_true(x): 真实的组织信号强度
- noise: 噪声

可视化方法：
1. 差分法：I_observed / I_corrected
2. 对数域：log(I_observed) - log(I_corrected)
3. 滤波法：低通滤波提取偏场场
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, restoration
import os
from pathlib import Path
import matplotlib

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
    except:
        pass

def visualize_bias_field(original_image, corrected_image, save_path=None,
                        method='division', slice_idx=None):
    """
    可视化MRI偏场场校正效果

    参数:
        original_image (numpy.ndarray): 原始的MRI图像（可能有偏场场）
        corrected_image (numpy.ndarray): 校正后的MRI图像
        save_path (str): 保存路径
        method (str): 偏场场估计方法 ('division', 'log_diff', 'filter')
        slice_idx (int or None): 要可视化的切片索引（None表示中间切片）

    返回:
        dict: 可视化结果和统计信息
    """

    # 确保输入是numpy数组
    if not isinstance(original_image, np.ndarray) or not isinstance(corrected_image, np.ndarray):
        raise ValueError("输入必须是numpy数组")

    # 检查形状一致性
    if original_image.shape != corrected_image.shape:
        raise ValueError("原始图像和校正图像的形状必须一致")

    print(f"输入图像形状: {original_image.shape}")
    print(f"原始图像强度范围: [{np.min(original_image):.2f}, {np.max(original_image):.2f}]")
    print(f"校正图像强度范围: [{np.min(corrected_image):.2f}, {np.max(corrected_image):.2f}]")

    # 选择切片（如果是3D图像）
    if len(original_image.shape) == 3:
        if slice_idx is None:
            slice_idx = original_image.shape[0] // 2
        print(f"选择切片: {slice_idx}")

        orig_slice = original_image[slice_idx]
        corr_slice = corrected_image[slice_idx]
    else:
        orig_slice = original_image
        corr_slice = corrected_image

    # 估计偏场场
    if method == 'division':
        # 除法方法：B(x) = I_original / I_corrected
        bias_field = orig_slice / (corr_slice + 1e-6)
        bias_field_log = np.log(bias_field + 1e-6)
        method_name = "除法方法"

    elif method == 'log_diff':
        # 对数差分方法：log(B(x)) = log(I_original) - log(I_corrected)
        bias_field_log = np.log(orig_slice + 1e-6) - np.log(corr_slice + 1e-6)
        bias_field = np.exp(bias_field_log)
        method_name = "对数差分方法"

    elif method == 'filter':
        # 滤波方法：对原始图像进行低通滤波
        bias_field = filters.gaussian(orig_slice, sigma=20, preserve_range=True)
        bias_field = bias_field / np.mean(bias_field)  # 归一化
        bias_field_log = np.log(bias_field + 1e-6)
        method_name = "滤波方法"

    else:
        raise ValueError(f"未知的方法: {method}")

    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行：图像显示
    # 原始图像 Original Image
    im1 = axes[0, 0].imshow(orig_slice, cmap='gray')
    axes[0, 0].set_title('原始图像 Original Image\n(有偏场场 With Bias Field)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 偏场场 Bias Field
    im2 = axes[0, 1].imshow(bias_field, cmap='hot')
    if method_name == "除法方法":
        method_en = "Division Method"
    elif method_name == "对数差分方法":
        method_en = "Log Difference Method"
    else:
        method_en = "Filter Method"
    axes[0, 1].set_title(f'估计的偏场场 Estimated Bias Field\n({method_name} / {method_en})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 校正后图像 Corrected Image
    im3 = axes[0, 2].imshow(corr_slice, cmap='gray')
    axes[0, 2].set_title('校正后图像 Corrected Image')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # 第二行：分析图 Second Row: Analysis
    # 偏场场对数显示 Bias Field (Log Scale)
    im4 = axes[1, 0].imshow(bias_field_log, cmap='RdBu_r',
                           vmin=-np.max(np.abs(bias_field_log)),
                           vmax=np.max(np.abs(bias_field_log)))
    axes[1, 0].set_title('偏场场 Bias Field\n(对数尺度 Log Scale)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # 强度分布直方图 Intensity Distribution Histogram
    axes[1, 1].hist(orig_slice.flatten(), bins=100, alpha=0.7,
                   color='blue', label='原始图像 Original', density=True)
    axes[1, 1].hist(corr_slice.flatten(), bins=100, alpha=0.7,
                   color='red', label='校正图像 Corrected', density=True)
    axes[1, 1].set_title('强度分布对比\nIntensity Distribution Comparison')
    axes[1, 1].set_xlabel('信号强度 Signal Intensity')
    axes[1, 1].set_ylabel('密度 Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 偏场场剖面线 Bias Field Profile
    center_y, center_x = orig_slice.shape[0] // 2, orig_slice.shape[1] // 2
    horizontal_orig = orig_slice[center_y, :]
    horizontal_corr = corr_slice[center_y, :]
    horizontal_bias = bias_field[center_y, :]

    axes[1, 2].plot(horizontal_orig, 'b-', label='原始图像 Original', alpha=0.7)
    axes[1, 2].plot(horizontal_corr, 'r-', label='校正图像 Corrected', alpha=0.7)
    axes[1, 2].plot(horizontal_bias * np.mean(horizontal_orig), 'g--',
                    label='偏场场×均值 Bias Field×Mean', alpha=0.7)
    axes[1, 2].set_title('水平剖面线对比\nHorizontal Profile Comparison')
    axes[1, 2].set_xlabel('像素位置 Pixel Position')
    axes[1, 2].set_ylabel('信号强度 Signal Intensity')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'MRI偏场场可视化分析 MRI Bias Field Visualization Analysis - 切片 Slice {slice_idx}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 创建输出文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")

    # 展示2秒后自动关闭
    plt.pause(2)
    plt.close()

    # 计算统计信息
    stats = analyze_bias_field(original_image, corrected_image, bias_field)

    return {
        'bias_field': bias_field,
        'bias_field_log': bias_field_log,
        'method': method,
        'slice_idx': slice_idx,
        'stats': stats
    }

def analyze_bias_field(original_image, corrected_image, bias_field):
    """
    分析偏场场的统计特征

    参数:
        original_image (numpy.ndarray): 原始图像
        corrected_image (numpy.ndarray): 校正图像
        bias_field (numpy.ndarray): 估计的偏场场

    返回:
        dict: 统计分析结果
    """
    stats = {}

    # 原始图像统计
    stats['original'] = {
        'mean': np.mean(original_image),
        'std': np.std(original_image),
        'cv': np.std(original_image) / np.mean(original_image),  # 变异系数
        'min': np.min(original_image),
        'max': np.max(original_image),
        'range': np.max(original_image) - np.min(original_image)
    }

    # 校正图像统计
    stats['corrected'] = {
        'mean': np.mean(corrected_image),
        'std': np.std(corrected_image),
        'cv': np.std(corrected_image) / np.mean(corrected_image),
        'min': np.min(corrected_image),
        'max': np.max(corrected_image),
        'range': np.max(corrected_image) - np.min(corrected_image)
    }

    # 偏场场统计
    stats['bias_field'] = {
        'mean': np.mean(bias_field),
        'std': np.std(bias_field),
        'cv': np.std(bias_field) / np.mean(bias_field),
        'min': np.min(bias_field),
        'max': np.max(bias_field),
        'range': np.max(bias_field) - np.min(bias_field),
        'symmetry': np.mean(bias_field)  # 理想情况下应该接近1.0
    }

    # 校正效果评估
    stats['improvement'] = {
        'cv_reduction': (stats['original']['cv'] - stats['corrected']['cv']) / stats['original']['cv'],
        'std_reduction': (stats['original']['std'] - stats['corrected']['std']) / stats['original']['std'],
        'range_reduction': (stats['original']['range'] - stats['corrected']['range']) / stats['original']['range']
    }

    # 相关性分析
    if len(original_image.shape) == 3:
        # 对于3D图像，分析中间切片
        mid_slice = original_image.shape[0] // 2
        orig_flat = original_image[mid_slice].flatten()
        corr_flat = corrected_image[mid_slice].flatten()
    else:
        orig_flat = original_image.flatten()
        corr_flat = corrected_image.flatten()

    correlation = np.corrcoef(orig_flat, corr_flat)[0, 1]
    stats['correlation'] = correlation

    # 打印统计信息
    print(f"\n偏场场分析统计:")
    print(f"原始图像 - 均值: {stats['original']['mean']:.2f}, "
          f"标准差: {stats['original']['std']:.2f}, "
          f"变异系数: {stats['original']['cv']:.3f}")
    print(f"校正图像 - 均值: {stats['corrected']['mean']:.2f}, "
          f"标准差: {stats['corrected']['std']:.2f}, "
          f"变异系数: {stats['corrected']['cv']:.3f}")
    print(f"偏场场 - 均值: {stats['bias_field']['mean']:.3f}, "
          f"标准差: {stats['bias_field']['std']:.3f}, "
          f"范围: [{stats['bias_field']['min']:.3f}, {stats['bias_field']['max']:.3f}]")
    print(f"校正效果 - CV减少: {stats['improvement']['cv_reduction']*100:.1f}%, "
          f"相关系数: {correlation:.3f}")

    return stats

def generate_synthetic_mri_with_bias(shape=(256, 256), bias_strength=0.3):
    """
    生成含有偏场场的合成MRI数据

    参数:
        shape (tuple): 图像形状
        bias_strength (float): 偏场场强度 (0.0-1.0)

    返回:
        tuple: (原始图像, 含偏场场的图像, 真实偏场场)
    """
    print(f"生成合成MRI数据 (形状: {shape}, 偏场强度: {bias_strength})")

    # 创建坐标网格
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    y, x = y / shape[0], x / shape[1]  # 归一化到[0, 1]

    # 生成真实的MRI解剖结构
    # 背景脑组织
    true_image = np.zeros(shape, dtype=float)

    # 添加主要的解剖结构（模拟脑部MRI）
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # 灰质
    gray_matter_radius = min(shape) // 3
    y_dist, x_dist = y - center_y/shape[0], x - center_x/shape[1]
    gray_matter_mask = (y_dist**2 + x_dist**2) <= (gray_matter_radius/min(shape))**2
    true_image[gray_matter_mask] = 0.6

    # 白质
    white_matter_radius = min(shape) // 4
    white_matter_mask = (y_dist**2 + x_dist**2) <= (white_matter_radius/min(shape))**2
    true_image[white_matter_mask] = 0.8

    # 脑室（CSF）
    ventricle_centers = [(0.5, 0.3), (0.5, 0.7)]
    for vy, vx in ventricle_centers:
        y_v, x_v = y - vy, x - vx
        ventricle_mask = (y_v**2 + x_v**2) <= 0.02**2
        true_image[ventricle_mask] = 0.2

    # 添加噪声
    noise = np.random.normal(0, 0.05, shape)
    true_image += noise
    true_image = np.clip(true_image, 0, 1)

    # 生成偏场场（平滑的空间变化）
    # 使用多个正弦波组合创建复杂的偏场场模式
    bias_field = np.ones(shape)

    # 添加不同频率和方向的偏场分量
    # 低频全局偏场
    bias_field += bias_strength * 0.3 * np.sin(2 * np.pi * y)
    bias_field += bias_strength * 0.3 * np.cos(2 * np.pi * x)

    # 中等频率偏场
    bias_field += bias_strength * 0.2 * np.sin(4 * np.pi * y * x)
    bias_field += bias_strength * 0.2 * np.cos(3 * np.pi * (y - x))

    # 高频局部偏场
    bias_field += bias_strength * 0.1 * np.sin(8 * np.pi * y) * np.cos(6 * np.pi * x)

    # 应用高斯平滑使偏场场更加真实
    bias_field = ndimage.gaussian_filter(bias_field, sigma=15)

    # 确保偏场场均值为1（保持整体强度水平）
    bias_field = bias_field / np.mean(bias_field)

    # 应用偏场场到真实图像
    biased_image = true_image * bias_field

    # 添加一些额外的噪声
    biased_image += np.random.normal(0, 0.02, shape)
    biased_image = np.clip(biased_image, 0, 1)

    print(f"合成数据生成完成")
    print(f"真实图像强度范围: [{np.min(true_image):.3f}, {np.max(true_image):.3f}]")
    print(f"偏场场范围: [{np.min(bias_field):.3f}, {np.max(bias_field):.3f}]")
    print(f"含偏场图像强度范围: [{np.min(biased_image):.3f}, {np.max(biased_image):.3f}]")

    return true_image, biased_image, bias_field

def simulate_bias_correction(biased_image, method='gaussian'):
    """
    模拟偏场场校正算法

    参数:
        biased_image (numpy.ndarray): 含偏场场的图像
        method (str): 校正方法 ('gaussian', 'homomorphic', 'polynomial')

    返回:
        numpy.ndarray: 校正后的图像
    """
    print(f"模拟偏场场校正 (方法: {method})")

    if method == 'gaussian':
        # 高斯滤波方法（简化版）
        # 估计偏场场
        estimated_bias = filters.gaussian(biased_image, sigma=20, preserve_range=True)
        estimated_bias = estimated_bias / np.mean(estimated_bias)

        # 校正图像
        corrected = biased_image / (estimated_bias + 1e-6)

    elif method == 'homomorphic':
        # 同态滤波方法（简化版）
        log_image = np.log(biased_image + 1e-6)

        # 高通滤波
        low_freq = filters.gaussian(log_image, sigma=20)
        high_freq = log_image - low_freq

        # 重构图像
        corrected_log = high_freq + np.mean(low_freq)
        corrected = np.exp(corrected_log)

    elif method == 'polynomial':
        # 多项式拟合方法（简化版）
        y, x = np.mgrid[0:biased_image.shape[0], 0:biased_image.shape[1]]
        y, x = y / biased_image.shape[0], x / biased_image.shape[1]

        # 拟合2D多项式偏场场
        # 这里使用简单的二次多项式
        from sklearn.linear_model import LinearRegression

        # 构造特征矩阵
        X = np.column_stack([
            np.ones(biased_image.size),
            y.flatten(), x.flatten(),
            y.flatten()**2, x.flatten()**2,
            y.flatten() * x.flatten()
        ])

        # 拟合偏场场
        model = LinearRegression()
        model.fit(X, biased_image.flatten())

        # 估计偏场场
        estimated_bias = model.predict(X).reshape(biased_image.shape)
        estimated_bias = estimated_bias / np.mean(estimated_bias)

        # 校正图像
        corrected = biased_image / (estimated_bias + 1e-6)

    else:
        raise ValueError(f"未知的校正方法: {method}")

    # 归一化到合理范围
    corrected = np.clip(corrected, 0, 1)

    return corrected

def demonstrate_different_methods():
    """
    演示不同偏场场估计和校正方法
    """
    print("\n" + "="*60)
    print("不同偏场场方法对比演示")
    print("="*60)

    # 生成合成数据
    true_image, biased_image, true_bias_field = generate_synthetic_mri_with_bias(
        shape=(256, 256), bias_strength=0.4
    )

    # 不同的校正方法
    correction_methods = ['gaussian', 'homomorphic', 'polynomial']
    results = {}

    for method in correction_methods:
        print(f"\n测试方法: {method}")

        # 模拟校正
        corrected_image = simulate_bias_correction(biased_image, method=method)

        # 可视化结果
        os.makedirs("output", exist_ok=True)
        save_path = f"output/bias_field_{method}_comparison.png"

        # 使用三种不同的偏场场估计方法进行可视化
        visualization_methods = ['division', 'log_diff', 'filter']

        for viz_method in visualization_methods:
            try:
                result = visualize_bias_field(
                    biased_image, corrected_image,
                    save_path=f"output/bias_field_{method}_{viz_method}.png",
                    method=viz_method
                )
                results[f"{method}_{viz_method}"] = result
            except Exception as e:
                print(f"可视化失败 {method}_{viz_method}: {e}")

    # 对比所有方法
    compare_all_methods(true_image, biased_image, true_bias_field, correction_methods)

def compare_all_methods(true_image, biased_image, true_bias_field, methods):
    """
    对比所有偏场场校正方法的效果
    """
    print("\n生成方法对比图...")

    fig, axes = plt.subplots(2, len(methods) + 2, figsize=(6*(len(methods)+2), 12))

    # 第一行：原始图像和偏场场
    axes[0, 0].imshow(true_image, cmap='gray')
    axes[0, 0].set_title('真实图像')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(biased_image, cmap='gray')
    axes[0, 1].set_title('含偏场图像')
    axes[0, 1].axis('off')

    # 第二行：偏场场
    axes[1, 0].imshow(true_bias_field, cmap='hot')
    axes[1, 0].set_title('真实偏场场')
    axes[1, 0].axis('off')

    # 估计偏场场（使用高斯滤波）
    estimated_bias = filters.gaussian(biased_image, sigma=20, preserve_range=True)
    estimated_bias = estimated_bias / np.mean(estimated_bias)
    axes[1, 1].imshow(estimated_bias, cmap='hot')
    axes[1, 1].set_title('估计偏场场')
    axes[1, 1].axis('off')

    # 各种校正方法
    for i, method in enumerate(methods):
        corrected_image = simulate_bias_correction(biased_image, method=method)

        # 校正后图像
        axes[0, i+2].imshow(corrected_image, cmap='gray')
        axes[0, i+2].set_title(f'{method.capitalize()} 校正')
        axes[0, i+2].axis('off')

        # 残差图像
        residual = np.abs(corrected_image - true_image)
        axes[1, i+2].imshow(residual, cmap='hot')
        axes[1, i+2].set_title(f'{method.capitalize()} 残差')
        axes[1, i+2].axis('off')

        # 计算评估指标
        mse = np.mean((corrected_image - true_image)**2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        ssim = calculate_ssim(corrected_image, true_image)

        print(f"Method {i+1} - MSE: {mse:.4f}, PSNR: {psnr:.1f}, SSIM: {ssim:.3f}")

    plt.suptitle('MRI偏场场校正方法对比', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = "output/bias_field_methods_comparison.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.pause(2)
    plt.close()

def calculate_ssim(img1, img2):
    """
    计算结构相似性指数（简化版）
    """
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=1.0)

def main():
    """
    主函数：演示偏场场可视化的完整流程
    """
    print("MRI偏场场可视化演示")
    print("="*50)

    try:
        # 生成合成MRI数据
        true_image, biased_image, true_bias_field = generate_synthetic_mri_with_bias(
            shape=(256, 256), bias_strength=0.4
        )

        # 模拟偏场场校正
        corrected_image = simulate_bias_correction(biased_image, method='gaussian')

        # 可视化偏场场校正效果
        os.makedirs("output", exist_ok=True)

        # 使用不同方法可视化偏场场
        methods = ['division', 'log_diff', 'filter']
        for method in methods:
            print(f"\n使用 {method} 方法可视化偏场场...")

            result = visualize_bias_field(
                biased_image, corrected_image,
                save_path=f"output/bias_field_visualization_{method}.png",
                method=method
            )

        # 演示不同校正方法
        demonstrate_different_methods()

        print("\n" + "="*60)
        print("总结")
        print("="*60)
        print("1. 偏场场是MRI图像中的常见问题")
        print("2. 多种方法可用于偏场场估计和可视化")
        print("3. 不同校正方法适用于不同类型的偏场场")
        print("4. 定量评估指标有助于方法选择")
        print("5. 偏场场校正能显著改善图像质量")

    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()