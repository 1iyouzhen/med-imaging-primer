#!/usr/bin/env python3
"""
CLAHE对比度增强
实现X射线图像的CLAHE（对比度限制自适应直方图均衡化）增强算法

学习目标：
1. 理解CLAHE算法的原理和优势
2. 掌握CLAHE的实现方法
3. 了解自适应参数调整策略

算法原理：
CLAHE (Contrast Limited Adaptive Histogram Equalization) 是改进的直方图均衡化算法：

1. **分块处理**: 将图像划分为小块 (如8×8)
2. **局部直方图均衡化**: 对每个块独立进行直方图均衡化
3. **对比度限制**: 限制直方图峰值，避免噪声放大
4. **双线性插值**: 块边界使用双线性插值平滑过渡

算法步骤：
1. 将图像分割为tiles
2. 计算每个tile的直方图
3. 限制直方图幅度 (clip_limit)
4. 重分布被限制的像素
5. 计算累积分布函数 (CDF)
6. 应用双线性插值
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
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

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE对比度增强

    参数:
        image (numpy.ndarray): 输入图像
        clip_limit (float): 对比度限制参数
        tile_grid_size (tuple): 分块大小 (rows, cols)

    返回:
        numpy.ndarray: 增强后的图像
    """
    # 确保输入是有效的numpy数组
    if not isinstance(image, np.ndarray):
        raise ValueError("输入必须是numpy数组")

    print(f"输入图像形状: {image.shape}")
    print(f"输入图像数据类型: {image.dtype}")
    print(f"输入图像范围: [{np.min(image):.2f}, {np.max(image):.2f}]")

    # 确保输入是8位图像
    if image.dtype != np.uint8:
        # 归一化并转换到8位
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image_8bit = image_normalized.astype(np.uint8)
        print("图像已转换为8位格式")
    else:
        image_8bit = image.copy()

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 应用CLAHE
    enhanced_image = clahe.apply(image_8bit)

    print(f"输出图像范围: [{np.min(enhanced_image):.2f}, {np.max(enhanced_image):.2f}]")

    return enhanced_image

def adaptive_clahe_parameters(image):
    """
    根据图像特征自适应调整CLAHE参数

    参数:
        image (numpy.ndarray): 输入图像

    返回:
        dict: 推荐的CLAHE参数
    """
    # 计算图像统计特征
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    dynamic_range = np.max(image) - np.min(image)

    # 计算图像对比度 (Michelson对比度)
    contrast = (np.max(image) - np.min(image)) / (np.max(image) + np.min(image))

    # 计算直方图特征
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist_normalized = hist / np.sum(hist)

    # 计算直方图的偏度和峰度
    pixel_values = image.flatten()
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)
    skewness = np.mean(((pixel_values - mean_val) / std_val) ** 3)
    kurtosis = np.mean(((pixel_values - mean_val) / std_val) ** 4) - 3

    # 自适应参数调整规则
    if dynamic_range < 50:  # 低对比度图像
        clip_limit = 3.0
        tile_size = (16, 16)
        enhancement_type = "低对比度增强"
    elif mean_intensity < 80:  # 暗图像
        clip_limit = 2.5
        tile_size = (12, 12)
        enhancement_type = "暗图像增强"
    elif mean_intensity > 180:  # 亮图像
        clip_limit = 2.0
        tile_size = (8, 8)
        enhancement_type = "亮图像增强"
    elif contrast < 0.3:  # 低对比度
        clip_limit = 3.0
        tile_size = (16, 16)
        enhancement_type = "低对比度增强"
    elif skewness > 0.5:  # 偏亮分布
        clip_limit = 2.2
        tile_size = (10, 10)
        enhancement_type = "偏亮分布校正"
    elif skewness < -0.5:  # 偏暗分布
        clip_limit = 2.8
        tile_size = (14, 14)
        enhancement_type = "偏暗分布校正"
    else:  # 正常图像
        clip_limit = 2.0
        tile_size = (8, 8)
        enhancement_type = "标准增强"

    parameters = {
        'clip_limit': clip_limit,
        'tile_size': tile_size,
        'enhancement_type': enhancement_type,
        'image_stats': {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'dynamic_range': dynamic_range,
            'contrast': contrast,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    }

    print(f"图像分析结果:")
    print(f"  平均亮度: {mean_intensity:.1f}")
    print(f"  动态范围: {dynamic_range:.1f}")
    print(f"  对比度: {contrast:.3f}")
    print(f"  偏度: {skewness:.3f}")
    print(f"  推荐增强类型: {enhancement_type}")
    print(f"  推荐参数: clip_limit={clip_limit}, tile_size={tile_size}")

    return parameters

def generate_synthetic_xray_image(shape=(512, 512)):
    """
    生成合成的X射线图像用于测试

    参数:
        shape (tuple): 图像形状

    返回:
        numpy.ndarray: 合成的X射线图像
    """
    print(f"生成合成X射线图像 (形状: {shape})")

    # 创建基础图像
    image = np.zeros(shape, dtype=np.uint8)

    # 添加背景结构
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # 模拟胸部轮廓
    chest_ellipse = ((x - center_x) / (shape[1] * 0.35))**2 + ((y - center_y) / (shape[0] * 0.45))**2 <= 1
    image[chest_ellipse] = 120

    # 模拟心脏
    heart_center = (center_y - 50, center_x + 50)
    heart_mask = ((x - heart_center[1]) / 80)**2 + ((y - heart_center[0]) / 100)**2 <= 1
    image[heart_mask] = 100

    # 模拟肺部（较暗区域）
    lung_left_center = (center_y - 80, center_x - 100)
    lung_right_center = (center_y - 80, center_x + 100)

    lung_left_mask = ((x - lung_left_center[1]) / 70)**2 + ((y - lung_left_center[0]) / 120)**2 <= 1
    lung_right_mask = ((x - lung_right_center[1]) / 70)**2 + ((y - lung_right_center[0]) / 120)**2 <= 1

    image[lung_left_mask] = 60
    image[lung_right_mask] = 60

    # 模拟肋骨
    for i in range(8):
        angle = i * np.pi / 8
        for radius in range(50, 200, 30):
            rib_y = int(center_y + radius * np.sin(angle))
            rib_x = int(center_x + radius * np.cos(angle))
            if 0 <= rib_y < shape[0] and 0 <= rib_x < shape[1]:
                # 绘制弧形肋骨
                for r in range(-5, 6):
                    y_pos = rib_y + int(r * np.cos(angle))
                    x_pos = rib_x - int(r * np.sin(angle))
                    if 0 <= y_pos < shape[0] and 0 <= x_pos < shape[1]:
                        image[y_pos, x_pos] = 140

    # 添加一些病理特征（模拟结节）
    nodule_pos = (center_y - 100, center_x - 80)
    nodule_mask = ((x - nodule_pos[1]) / 15)**2 + ((y - nodule_pos[0]) / 15)**2 <= 1
    image[nodule_mask] = 80

    # 添加噪声和纹理
    noise = np.random.normal(0, 10, shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 应用高斯模糊使图像更自然
    image = cv2.GaussianBlur(image, (3, 3), 0)

    print(f"合成图像生成完成")
    print(f"图像强度范围: [{np.min(image):.0f}, {np.max(image):.0f}]")

    return image

def compare_clahe_parameters(image, save_path=None):
    """
    比较不同CLAHE参数的效果

    参数:
        image (numpy.ndarray): 输入图像
        save_path (str): 保存路径
    """
    print("比较不同CLAHE参数的效果...")

    # 定义不同的参数组合
    parameter_sets = [
        {'clip_limit': 1.0, 'tile_size': (8, 8), 'name': '弱增强'},
        {'clip_limit': 2.0, 'tile_size': (8, 8), 'name': '标准增强'},
        {'clip_limit': 3.0, 'tile_size': (8, 8), 'name': '强增强'},
        {'clip_limit': 2.0, 'tile_size': (4, 4), 'name': '小块增强'},
        {'clip_limit': 2.0, 'tile_size': (16, 16), 'name': '大块增强'},
        {'clip_limit': 4.0, 'tile_size': (16, 16), 'name': '最强增强'}
    ]

    # 创建对比图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 原始图像
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    # 原始直方图
    hist, bins, _ = axes[1, 0].hist(image.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_title('原始直方图')
    axes[1, 0].set_xlabel('像素值')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].grid(True, alpha=0.3)

    # 应用不同参数的CLAHE
    enhanced_images = []
    for i, params in enumerate(parameter_sets[:3]):  # 第一行显示前3个
        enhanced = clahe_enhancement(image, params['clip_limit'], params['tile_size'])
        enhanced_images.append(enhanced)

        axes[0, i+1].imshow(enhanced, cmap='gray')
        axes[0, i+1].set_title(f"{params['name']}\nclip={params['clip_limit']}, tile={params['tile_size']}")
        axes[0, i+1].axis('off')

        # 直方图
        axes[1, i+1].hist(enhanced.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, i+1].set_title(f"{params['name']}直方图")
        axes[1, i+1].set_xlabel('像素值')
        axes[1, i+1].set_ylabel('频次')
        axes[1, i+1].grid(True, alpha=0.3)

    plt.suptitle('CLAHE参数对比', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"参数对比图已保存至: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

    return enhanced_images

def visualize_clahe_process(image, enhanced_image, parameters=None, save_path=None):
    """
    可视化CLAHE处理过程的详细分析

    参数:
        image (numpy.ndarray): 原始图像
        enhanced_image (numpy.ndarray): 增强后图像
        parameters (dict): CLAHE参数
        save_path (str): 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 第一行：图像对比
    # 原始图像
    im1 = axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 增强后图像
    im2 = axes[0, 1].imshow(enhanced_image, cmap='gray')
    axes[0, 1].set_title('CLAHE增强后图像')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 差异图像
    diff_image = enhanced_image.astype(np.float32) - image.astype(np.float32)
    im3 = axes[0, 2].imshow(diff_image, cmap='RdBu_r',
                           vmin=-50, vmax=50)
    axes[0, 2].set_title('差异图像 (增强-原始)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # 第二行：分析图
    # 直方图对比
    axes[1, 0].hist(image.flatten(), bins=50, alpha=0.7, color='blue',
                   label='原始', density=True)
    axes[1, 0].hist(enhanced_image.flatten(), bins=50, alpha=0.7, color='red',
                   label='增强', density=True)
    axes[1, 0].set_title('直方图对比')
    axes[1, 0].set_xlabel('像素值')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 累积分布函数对比
    orig_sorted = np.sort(image.flatten())
    enh_sorted = np.sort(enhanced_image.flatten())
    orig_cdf = np.arange(len(orig_sorted)) / len(orig_sorted)
    enh_cdf = np.arange(len(enh_sorted)) / len(enh_sorted)

    axes[1, 1].plot(orig_sorted, orig_cdf, 'b-', label='原始CDF', alpha=0.7)
    axes[1, 1].plot(enh_sorted, enh_cdf, 'r-', label='增强CDF', alpha=0.7)
    axes[1, 1].set_title('累积分布函数对比')
    axes[1, 1].set_xlabel('像素值')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 局部对比度分析
    # 计算局部对比度（使用局部标准差）
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    orig_local_var = cv2.filter2D(image.astype(np.float32), -1, kernel)
    enh_local_var = cv2.filter2D(enhanced_image.astype(np.float32), -1, kernel)

    # 选择中心线进行对比度分析
    center_line = image.shape[0] // 2
    orig_profile = image[center_line, :]
    enh_profile = enhanced_image[center_line, :]

    axes[1, 2].plot(orig_profile, 'b-', label='原始', alpha=0.7)
    axes[1, 2].plot(enh_profile, 'r-', label='增强', alpha=0.7)
    axes[1, 2].set_title(f'水平剖面线对比 (行 {center_line})')
    axes[1, 2].set_xlabel('列位置')
    axes[1, 2].set_ylabel('像素值')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # 添加参数信息
    param_text = ""
    if parameters:
        param_text = f"CLAHE参数: clip_limit={parameters.get('clip_limit', 'N/A')}, "
        param_text += f"tile_size={parameters.get('tile_size', 'N/A')}"

    plt.suptitle(f'CLAHE增强效果分析\n{param_text}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"详细分析图已保存至: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

def calculate_enhancement_metrics(original_image, enhanced_image):
    """
    计算图像增强效果的定量指标

    参数:
        original_image (numpy.ndarray): 原始图像
        enhanced_image (numpy.ndarray): 增强后图像

    返回:
        dict: 增强效果指标
    """
    metrics = {}

    # 1. 对比度指标 (使用标准差)
    orig_contrast = np.std(original_image)
    enh_contrast = np.std(enhanced_image)
    metrics['contrast_improvement'] = enh_contrast / orig_contrast

    # 2. 动态范围指标
    orig_range = np.max(original_image) - np.min(original_image)
    enh_range = np.max(enhanced_image) - np.min(enhanced_image)
    metrics['range_expansion'] = enh_range / orig_range

    # 3. 熵指标 (信息量)
    from skimage import filters
    orig_entropy = filters.rank.entropy(original_image, np.ones((7, 7)))
    enh_entropy = filters.rank.entropy(enhanced_image, np.ones((7, 7)))
    metrics['entropy_improvement'] = np.mean(enh_entropy) / np.mean(orig_entropy)

    # 4. 边缘强度指标
    orig_edges = cv2.Canny(original_image, 50, 150)
    enh_edges = cv2.Canny(enhanced_image, 50, 150)
    metrics['edge_strength_improvement'] = np.sum(enh_edges) / np.sum(orig_edges)

    # 5. 峰值信噪比 (PSNR)
    mse = np.mean((original_image - enhanced_image) ** 2)
    if mse > 0:
        metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        metrics['psnr'] = float('inf')

    # 6. 结构相似性 (SSIM)
    try:
        from skimage.metrics import structural_similarity
        metrics['ssim'] = structural_similarity(original_image, enhanced_image)
    except:
        metrics['ssim'] = None

    print("增强效果定量评估:")
    print(f"  对比度提升倍数: {metrics['contrast_improvement']:.2f}")
    print(f"  动态范围扩展倍数: {metrics['range_expansion']:.2f}")
    print(f"  信息量提升倍数: {metrics['entropy_improvement']:.2f}")
    print(f"  边缘强度提升倍数: {metrics['edge_strength_improvement']:.2f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    if metrics['ssim']:
        print(f"  SSIM: {metrics['ssim']:.3f}")

    return metrics

def main():
    """
    主函数：演示CLAHE对比度增强的完整流程
    """
    print("CLAHE对比度增强演示")
    print("="*50)

    try:
        # 生成合成的X射线图像
        original_image = generate_synthetic_xray_image(shape=(512, 512))

        # 自适应参数分析
        print("\n" + "="*50)
        print("自适应参数分析")
        print("="*50)

        parameters = adaptive_clahe_parameters(original_image)

        # 使用推荐参数进行CLAHE增强
        print(f"\n使用推荐参数进行CLAHE增强...")
        enhanced_image = clahe_enhancement(
            original_image,
            clip_limit=parameters['clip_limit'],
            tile_grid_size=parameters['tile_size']
        )

        # 比较不同参数效果
        print(f"\n比较不同CLAHE参数效果...")
        os.makedirs("output", exist_ok=True)
        comparison_path = "output/clahe_parameter_comparison.png"
        compare_clahe_parameters(original_image, comparison_path)

        # 详细效果分析
        print(f"\n详细效果分析...")
        analysis_path = "output/clahe_detailed_analysis.png"
        visualize_clahe_process(original_image, enhanced_image, parameters, analysis_path)

        # 定量评估
        print(f"\n定量评估增强效果...")
        metrics = calculate_enhancement_metrics(original_image, enhanced_image)

        print("\n" + "="*60)
        print("总结")
        print("="*60)
        print("1. CLAHE是有效的X射线图像对比度增强方法")
        print("2. 自适应参数选择能针对不同图像类型优化效果")
        print("3. clip_limit控制对比度增强强度")
        print("4. tile_size影响局部增强的粒度")
        print("5. 定量指标有助于客观评估增强效果")
        print("6. CLAHE能显著改善低对比度医学图像的质量")

    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()