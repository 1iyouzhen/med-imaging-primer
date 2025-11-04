#!/usr/bin/env python3
"""
White Stripe强度标准化算法实现
实现MRI图像的White Stripe强度标准化，包括完整的测试和可视化功能

学习目标：
1. 理解White Stripe算法的原理和临床意义
2. 掌握白质识别和强度标准化的方法
3. 了解不同MRI序列的参数调整策略

算法原理：
White Stripe是一种简单而有效的MRI强度标准化方法：

核心思想：
1. 在脑部MRI中，白质具有相对稳定的信号特征
2. 通过统计分析找到白质的主模态
3. 将白质范围映射到标准区间（如[0, 1]）

算法步骤：
1. 计算强度直方图
2. 寻找最高峰（通常是白质）
3. 确定白质强度范围
4. 创建白质mask
5. 计算白质强度统计
6. 线性标准化到[0, 1]

技术特点：
- 基于白质的稳定性
- 自适应阈值选择
- 模态特定参数
- 保持组织对比度
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from skimage import filters, morphology
import os
from pathlib import Path
import matplotlib

# 设置字体 - 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'Helvetica']

def white_stripe_normalization(image, modality='T1', width=None,
                              max_iterations=10, convergence_threshold=0.01):
    """
    White Stripe强度标准化

    参数:
        image (numpy.ndarray): 输入的MRI图像
        modality (str): MRI序列类型 ('T1', 'T2', 'FLAIR')
        width (float): 白质强度的宽度比例，如果为None则使用默认值
        max_iterations (int): 最大迭代次数
        convergence_threshold (float): 收敛阈值

    返回:
        tuple: (标准化图像, (下界, 上界), 统计信息)
    """
    print(f"White Stripe标准化开始")
    print(f"输入图像形状: {image.shape}")
    print(f"输入图像范围: [{np.min(image):.3f}, {np.max(image):.3f}]")
    print(f"MRI模态: {modality}")

    # 根据模态设置默认参数
    default_params = {
        'T1': {'width': 0.1, 'lower_percentile': 20, 'upper_percentile': 80},
        'T2': {'width': 0.05, 'lower_percentile': 15, 'upper_percentile': 85},
        'FLAIR': {'width': 0.1, 'lower_percentile': 25, 'upper_percentile': 75}
    }

    if modality not in default_params:
        print(f"未知模态 {modality}，使用T1默认参数")
        modality = 'T1'

    if width is None:
        width = default_params[modality]['width']

    print(f"使用参数: width={width}, max_iterations={max_iterations}")

    # 迭代优化白质范围
    lower_bound, upper_bound = find_white_stripe_range(
        image, modality, width, max_iterations, convergence_threshold
    )

    print(f"识别的白质范围: [{lower_bound:.3f}, {upper_bound:.3f}]")

    # 创建白质mask
    white_matter_mask = (image >= lower_bound) & (image <= upper_bound)
    white_matter_pixels = image[white_matter_mask]

    if len(white_matter_pixels) == 0:
        print("警告：未检测到白质像素，使用全局统计")
        white_matter_pixels = image.flatten()

    # 计算白质强度统计
    wm_mean = np.mean(white_matter_pixels)
    wm_std = np.std(white_matter_pixels)
    wm_median = np.median(white_matter_pixels)

    print(f"白质统计: 均值={wm_mean:.3f}, 标准差={wm_std:.3f}, 中位数={wm_median:.3f}")
    print(f"白质像素数量: {len(white_matter_pixels):,} ({len(white_matter_pixels)/image.size*100:.1f}%)")

    # 线性标准化到[0, 1]
    # 方法1: 基于白质均值和标准差
    normalized_image = (image - wm_mean) / wm_std
    normalized_image = np.clip(normalized_image, -3, 3)  # 限制极端值
    normalized_image = (normalized_image + 3) / 6  # 映射到[0, 1]

    # 计算统计信息
    stats = {
        'original_stats': {
            'mean': np.mean(image),
            'std': np.std(image),
            'min': np.min(image),
            'max': np.max(image),
            'median': np.median(image)
        },
        'normalized_stats': {
            'mean': np.mean(normalized_image),
            'std': np.std(normalized_image),
            'min': np.min(normalized_image),
            'max': np.max(normalized_image),
            'median': np.median(normalized_image)
        },
        'white_matter_stats': {
            'mean': wm_mean,
            'std': wm_std,
            'median': wm_median,
            'range': (lower_bound, upper_bound),
            'pixel_count': len(white_matter_pixels),
            'percentage': len(white_matter_pixels) / image.size * 100
        },
        'parameters': {
            'modality': modality,
            'width': width,
            'max_iterations': max_iterations,
            'convergence_threshold': convergence_threshold
        }
    }

    print(f"标准化完成，输出范围: [{np.min(normalized_image):.3f}, {np.max(normalized_image):.3f}]")

    return normalized_image, (lower_bound, upper_bound), stats

def find_white_stripe_range(image, modality, width, max_iterations, convergence_threshold):
    """
    迭代寻找最优的白质范围

    参数:
        image (numpy.ndarray): 输入图像
        modality (str): MRI模态
        width (float): 宽度比例
        max_iterations (int): 最大迭代次数
        convergence_threshold (float): 收敛阈值

    返回:
        tuple: (下界, 上界)
    """
    print("开始迭代寻找白质范围...")

    # 初始范围估计
    flat_image = image.flatten()

    # 移除明显的异常值
    q1, q99 = np.percentile(flat_image, [1, 99])
    filtered_image = flat_image[(flat_image >= q1) & (flat_image <= q99)]

    # 计算初始直方图
    hist, bin_edges = np.histogram(filtered_image, bins=256, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 找到最高峰
    peak_idx = np.argmax(hist)
    initial_peak = bin_centers[peak_idx]

    print(f"初始峰值: {initial_peak:.3f}")

    # 迭代优化
    lower_bound = initial_peak - width * initial_peak
    upper_bound = initial_peak + width * initial_peak

    for iteration in range(max_iterations):
        # 创建当前范围的白质mask
        mask = (image >= lower_bound) & (image <= upper_bound)
        white_pixels = image[mask]

        if len(white_pixels) == 0:
            print(f"迭代 {iteration + 1}: 无白质像素，停止迭代")
            break

        # 计算当前白质的统计
        current_mean = np.mean(white_pixels)
        current_median = np.median(white_pixels)

        # 根据模态选择参考值
        if modality == 'T1':
            reference = current_mean  # T1中白质通常是最高信号
        elif modality == 'T2':
            reference = current_median  # T2中用中位数更稳定
        else:  # FLAIR
            reference = current_mean

        # 更新范围
        new_lower = reference - width * reference
        new_upper = reference + width * reference

        # 计算变化
        change = abs(new_lower - lower_bound) + abs(new_upper - upper_bound)

        print(f"迭代 {iteration + 1}: 范围=[{new_lower:.3f}, {new_upper:.3f}], 变化={change:.6f}")

        # 检查收敛
        if change < convergence_threshold:
            print(f"收敛于迭代 {iteration + 1}")
            break

        lower_bound = new_lower
        upper_bound = new_upper

    return lower_bound, upper_bound

def generate_synthetic_mri_data(shape=(128, 128), modality='T1', noise_level=0.05,
                                bias_field_strength=0.2):
    """
    生成合成的MRI数据用于测试

    参数:
        shape (tuple): 图像形状
        modality (str): MRI模态
        noise_level (float): 噪声水平
        bias_field_strength (float): 偏场场强度

    返回:
        numpy.ndarray: 合成的MRI图像
    """
    print(f"生成合成{modality} MRI数据 (形状: {shape})")

    # 创建坐标网格
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    y, x = y / shape[0], x / shape[1]  # 归一化到[0, 1]

    # 生成基础解剖结构
    image = np.zeros(shape, dtype=np.float32)
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # 模拟脑部轮廓
    brain_mask = ((x - center_x/shape[1])**2 + (y - center_y/shape[0])**2) <= 0.25
    image[brain_mask] = 0.3  # 背景组织

    # 添加白质
    white_matter_centers = [
        (center_y/shape[0], center_x/shape[1]),  # 中央
        (center_y/shape[0] + 0.1, center_x/shape[1] - 0.1),  # 左侧
        (center_y/shape[0] + 0.1, center_x/shape[1] + 0.1),  # 右侧
    ]

    for cy, cx in white_matter_centers:
        wm_mask = ((x - cx)**2 + (y - cy)**2) <= 0.06
        if modality == 'T1':
            image[wm_mask] = 0.8  # T1白质高信号
        elif modality == 'T2':
            image[wm_mask] = 0.6  # T2白质中等信号
        else:  # FLAIR
            image[wm_mask] = 0.4  # FLAIR白质低信号

    # 添加灰质
    gray_matter_mask = brain_mask & (image == 0.3)
    if modality == 'T1':
        image[gray_matter_mask] = 0.5  # T1灰质中等信号
    elif modality == 'T2':
        image[gray_matter_mask] = 0.7  # T2灰质高信号
    else:  # FLAIR
        image[gray_matter_mask] = 0.9  # FLAIR灰质最高信号

    # 添加脑室（CSF）
    ventricle_centers = [(center_y/shape[0], center_x/shape[1])]
    for vy, vx in ventricle_centers:
        csf_mask = ((x - vx)**2 + (y - vy)**2) <= 0.02
        if modality == 'T1':
            image[csf_mask] = 0.1  # T1 CSF低信号
        elif modality == 'T2':
            image[csf_mask] = 0.9  # T2 CSF高信号
        else:  # FLAIR
            image[csf_mask] = 0.1  # FLAIR CSF低信号

    # 添加偏场场
    if bias_field_strength > 0:
        bias_field = np.ones(shape)
        bias_field += bias_field_strength * 0.3 * np.sin(2 * np.pi * y)
        bias_field += bias_field_strength * 0.2 * np.cos(2 * np.pi * x)
        bias_field = ndimage.gaussian_filter(bias_field, sigma=20)
        bias_field = bias_field / np.mean(bias_field)
        image *= bias_field

    # 添加噪声
    noise = np.random.normal(0, noise_level, shape)
    image += noise

    # 确保在合理范围内
    image = np.clip(image, 0, 1)

    print(f"合成数据生成完成")
    print(f"信号范围: [{np.min(image):.3f}, {np.max(image):.3f}]")

    return image

def visualize_white_stripe_normalization(original_image, normalized_image,
                                      white_range, stats, save_path=None):
    """
    可视化White Stripe标准化效果

    参数:
        original_image (numpy.ndarray): 原始图像
        normalized_image (numpy.ndarray): 标准化图像
        white_range (tuple): 白质范围
        stats (dict): 统计信息
        save_path (str): 保存路径
    """
    print("生成White Stripe标准化可视化...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行：图像显示
    # 原始图像
    im1 = axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 标准化图像
    im2 = axes[0, 1].imshow(normalized_image, cmap='gray')
    axes[0, 1].set_title('White Stripe标准化图像')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 差异图像
    diff = normalized_image - (original_image - np.mean(original_image)) / np.std(original_image)
    im3 = axes[0, 2].imshow(diff, cmap='RdBu_r',
                           vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[0, 2].set_title('差异图像')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # 第二行：分析图
    # 原始直方图
    axes[1, 0].hist(original_image.flatten(), bins=100, alpha=0.7, color='blue', density=True)
    axes[1, 0].axvline(white_range[0], color='red', linestyle='--', label=f'下界: {white_range[0]:.3f}')
    axes[1, 0].axvline(white_range[1], color='red', linestyle='--', label=f'上界: {white_range[1]:.3f}')
    axes[1, 0].axvline(stats['white_matter_stats']['mean'], color='green',
                       linestyle='-', label=f'白质均值: {stats["white_matter_stats"]["mean"]:.3f}')
    axes[1, 0].set_title('原始强度分布')
    axes[1, 0].set_xlabel('强度值')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 标准化直方图
    axes[1, 1].hist(normalized_image.flatten(), bins=100, alpha=0.7, color='red', density=True)
    axes[1, 1].set_title('标准化强度分布')
    axes[1, 1].set_xlabel('标准化强度')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].grid(True, alpha=0.3)

    # 统计对比
    stats_text = f"原始图像统计:\n"
    stats_text += f"  均值: {stats['original_stats']['mean']:.3f}\n"
    stats_text += f"  标准差: {stats['original_stats']['std']:.3f}\n"
    stats_text += f"  范围: [{stats['original_stats']['min']:.3f}, {stats['original_stats']['max']:.3f}]\n\n"
    stats_text += f"标准化图像统计:\n"
    stats_text += f"  均值: {stats['normalized_stats']['mean']:.3f}\n"
    stats_text += f"  标准差: {stats['normalized_stats']['std']:.3f}\n"
    # 在终端打印详细的统计信息
    print("\n" + "="*50)
    print("WHITE STRIPE标准化统计信息")
    print("="*50)
    print(f"输入图像信息:")
    print(f"  形状: {original_image.shape}")
    print(f"  范围: [{stats['original_stats']['min']:.3f}, {stats['original_stats']['max']:.3f}]")
    print(f"输出图像信息:")
    print(f"  范围: [{stats['normalized_stats']['min']:.3f}, {stats['normalized_stats']['max']:.3f}]")
    print(f"白质统计:")
    print(f"  像素数量: {stats['white_matter_stats']['pixel_count']:,}")
    print(f"  占比: {stats['white_matter_stats']['percentage']:.1f}%")
    print(f"  均值: {stats['white_matter_stats']['mean']:.3f}")
    print("="*50)
    # 在图表中只显示简化的统计信息
    simplified_stats = f"白质像素: {stats['white_matter_stats']['pixel_count']:,}"
    simplified_stats += f"\n占比: {stats['white_matter_stats']['percentage']:.1f}%"
    simplified_stats += f"\n标准化范围: [{stats['normalized_stats']['min']:.3f}, {stats['normalized_stats']['max']:.3f}]"

    print(f"关键统计信息: {simplified_stats}")
    axes[1, 2].set_title('关键统计信息')
    axes[1, 2].axis('off')

    plt.suptitle(f'White Stripe Normalization Analysis\nModality: {stats["parameters"]["modality"]}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    # 展示2秒后自动关闭
    plt.pause(2)
    plt.close()

def compare_different_modalities():
    """
    比较不同MRI模态的White Stripe标准化效果
    """
    print("比较不同MRI模态的White Stripe标准化效果...")

    modalities = ['T1', 'T2', 'FLAIR']
    results = {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, modality in enumerate(modalities):
        print(f"\n处理 {modality} 模态...")

        # 生成合成数据
        original_image = generate_synthetic_mri_data(
            shape=(128, 128), modality=modality, noise_level=0.05, bias_field_strength=0.2
        )

        # 执行White Stripe标准化
        normalized_image, white_range, stats = white_stripe_normalization(
            original_image, modality=modality
        )

        results[modality] = {
            'original': original_image,
            'normalized': normalized_image,
            'white_range': white_range,
            'stats': stats
        }

        # 显示原始图像
        axes[0, i].imshow(original_image, cmap='gray')
        axes[0, i].set_title(f'{modality} 原始图像')
        axes[0, i].axis('off')

        # 显示标准化图像
        axes[1, i].imshow(normalized_image, cmap='gray')
        axes[1, i].set_title(f'{modality} 标准化图像\n白质范围: [{white_range[0]:.2f}, {white_range[1]:.2f}]')
        axes[1, i].axis('off')

    plt.suptitle('不同MRI模态的White Stripe标准化对比', fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    save_path = "output/white_stripe_modality_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.pause(2)  # 展示2秒
    plt.close()

    # 打印对比统计
    print("\n模态对比统计:")
    for modality, result in results.items():
        stats = result['stats']
        print(f"\n{modality}:")
        print(f"  原始均值: {stats['original_stats']['mean']:.3f}")
        print(f"  标准化均值: {stats['normalized_stats']['mean']:.3f}")
        print(f"  白质像素: {stats['white_matter_stats']['pixel_count']:,} ({stats['white_matter_stats']['percentage']:.1f}%)")

def test_parameter_sensitivity():
    """
    测试参数敏感性
    """
    print("测试White Stripe参数敏感性...")

    # 生成测试数据
    test_image = generate_synthetic_mri_data(shape=(128, 128), modality='T1')

    # 不同的width参数
    widths = [0.05, 0.1, 0.15, 0.2]

    fig, axes = plt.subplots(2, len(widths), figsize=(6*len(widths), 12))

    for i, width in enumerate(widths):
        print(f"\n测试 width = {width}")

        # 执行标准化
        normalized_image, white_range, stats = white_stripe_normalization(
            test_image, modality='T1', width=width
        )

        # 显示标准化图像
        axes[0, i].imshow(normalized_image, cmap='gray')
        axes[0, i].set_title(f'Width = {width}\n标准化图像')
        axes[0, i].axis('off')

        # 显示直方图
        axes[1, i].hist(normalized_image.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        axes[1, i].set_title(f'Width = {width}\n强度分布')
        axes[1, i].set_xlabel('标准化强度')
        axes[1, i].set_ylabel('密度')
        axes[1, i].grid(True, alpha=0.3)

        print(f"  白质范围: [{white_range[0]:.3f}, {white_range[1]:.3f}]")
        print(f"  白质像素: {stats['white_matter_stats']['pixel_count']:,}")

    plt.suptitle('White Stripe参数敏感性测试', fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    save_path = "output/white_stripe_parameter_sensitivity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.pause(2)  # 展示2秒
    plt.close()

def main():
    """
    主函数：演示White Stripe标准化的完整流程
    """
    print("White Stripe强度标准化演示")
    print("="*50)

    try:
        # 演示T1模态
        print("\n" + "="*50)
        print("演示T1模态White Stripe标准化")
        print("="*50)

        # 生成T1合成数据
        t1_image = generate_synthetic_mri_data(shape=(128, 128), modality='T1',
                                              noise_level=0.05, bias_field_strength=0.3)

        # 执行White Stripe标准化
        normalized_t1, white_range_t1, stats_t1 = white_stripe_normalization(t1_image, modality='T1')

        # 可视化结果
        os.makedirs("output", exist_ok=True)
        save_path = "output/white_stripe_t1_normalization.png"
        visualize_white_stripe_normalization(t1_image, normalized_t1, white_range_t1, stats_t1, save_path)

        # 比较不同模态
        compare_different_modalities()

        # 测试参数敏感性
        test_parameter_sensitivity()

        print("\n" + "="*60)
        print("总结")
        print("="*60)
        print("1. White Stripe是有效的MRI强度标准化方法")
        print("2. 基于白质的稳定性进行标准化")
        print("3. 不同MRI模态需要不同的参数设置")
        print("4. 迭代优化能找到最优的白质范围")
        print("5. 标准化后图像具有一致的强度范围")
        print("6. 有助于后续的深度学习和定量分析")

    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()