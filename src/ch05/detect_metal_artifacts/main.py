#!/usr/bin/env python3
"""
CT金属伪影检测和处理
实现医学影像中金属伪影的自动检测和处理算法

学习目标：
1. 理解金属伪影的形成机理和特征
2. 掌握金属伪影检测的算法原理
3. 了解金属伪影处理的不同策略

算法原理：
金属伪影是CT扫描中常见的问题，主要表现为：
1. 高密度区域的极端HU值 (>3000 HU)
2. 条状伪影从金属区域向外辐射
3. 局部信号丢失和噪声增加

检测算法：
1. 阈值检测：基于HU值阈值识别金属区域
2. 连通性分析：去除孤立的噪声点
3. 形态学处理：优化金属区域的形状
4. 面积过滤：保留显著的金属植入物

处理策略：
1. 金属伪影校正 (Metal Artifact Reduction, MAR)
2. 替换插值：用周围组织替换金属区域
3. 迭代重建：使用特殊算法减少伪影
4. 深度学习：基于AI的伪影检测和校正
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure, filters
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

def detect_metal_artifacts(image, threshold=3000, min_area=100,
                          connectivity=2, morphological_operation=True):
    """
    检测金属伪影区域

    参数:
        image (numpy.ndarray): 输入的CT图像（3D或2D）
        threshold (float): HU值阈值，高于此值被认为是金属
        min_area (int): 最小面积阈值，过滤小的噪声区域
        connectivity (int): 连通性定义 (1: 4连通, 2: 8连通 for 2D; 1: 6连通, 2: 26连通 for 3D)
        morphological_operation (bool): 是否应用形态学操作

    返回:
        numpy.ndarray: 金属伪影区域的二值掩膜
        dict: 检测统计信息
    """
    # 确保输入是numpy数组
    if not isinstance(image, np.ndarray):
        raise ValueError("输入必须是numpy数组")

    # 记录原始图像信息
    original_shape = image.shape
    print(f"输入图像形状: {original_shape}")
    print(f"HU值范围: [{np.min(image):.1f}, {np.max(image):.1f}]")

    # 步骤1: 阈值检测 - 识别高密度区域
    metal_mask = image > threshold
    initial_metal_pixels = np.sum(metal_mask)
    print(f"阈值检测发现 {initial_metal_pixels:,} 个金属像素")

    if initial_metal_pixels == 0:
        print("未检测到金属伪影")
        return np.zeros_like(image, dtype=bool), {'metal_count': 0}

    # 步骤2: 连通性分析 - 识别独立的金属区域
    labeled_mask, num_features = ndimage.label(metal_mask,
                                               structure=np.ones((3,3,3) if len(image.shape) == 3 else (3,3)))
    print(f"发现 {num_features} 个独立的金属区域")

    # 步骤3: 面积过滤 - 去除小的噪声区域
    significant_metal = np.zeros_like(metal_mask, dtype=bool)
    region_stats = []

    for i in range(1, num_features + 1):
        region_mask = labeled_mask == i
        region_area = np.sum(region_mask)

        if region_area >= min_area:
            significant_metal |= region_mask
            region_stats.append({
                'region_id': i,
                'area': region_area,
                'mean_hu': np.mean(image[region_mask]),
                'max_hu': np.max(image[region_mask]),
                'centroid': ndimage.center_of_mass(region_mask)
            })

    print(f"面积过滤后保留 {len(region_stats)} 个金属区域")
    filtered_metal_pixels = np.sum(significant_metal)
    print(f"过滤后金属像素数量: {filtered_metal_pixels:,}")

    # 步骤4: 形态学操作 - 优化金属区域形状
    if morphological_operation and filtered_metal_pixels > 0:
        print(f"形态学操作前金属像素数量: {filtered_metal_pixels:,}")
        
        if len(image.shape) == 3:
            # 3D形态学操作 - 使用更小的结构元素
            structure = morphology.ball(radius=1)  # 减小半径
            # 只进行闭操作填充小孔，不开操作
            significant_metal = morphology.binary_closing(significant_metal, structure)
        else:
            # 2D形态学操作 - 使用更小的结构元素
            structure = morphology.disk(radius=1)  # 减小半径
            # 只进行闭操作填充小孔，不开操作
            significant_metal = morphology.binary_closing(significant_metal, structure)

        final_metal_pixels = np.sum(significant_metal)
        print(f"形态学操作后金属像素数量: {final_metal_pixels:,}")

    # 统计信息
    stats = {
        'metal_count': len(region_stats),
        'total_metal_pixels': np.sum(significant_metal),
        'metal_percentage': (np.sum(significant_metal) / image.size) * 100,
        'threshold': threshold,
        'min_area': min_area,
        'regions': region_stats,
        'initial_metal_pixels': initial_metal_pixels
    }

    return significant_metal, stats

def analyze_metal_artifacts(image, metal_mask, stats):
    """
    分析金属伪影的特征

    参数:
        image (numpy.ndarray): 原始CT图像
        metal_mask (numpy.ndarray): 金属区域掩膜
        stats (dict): 检测统计信息

    返回:
        dict: 详细的金属伪影分析
    """
    analysis = {}

    # 分析金属区域的HU值分布
    metal_hu_values = image[metal_mask]
    if len(metal_hu_values) > 0:
        analysis['metal_hu_stats'] = {
            'mean': np.mean(metal_hu_values),
            'std': np.std(metal_hu_values),
            'min': np.min(metal_hu_values),
            'max': np.max(metal_hu_values),
            'median': np.median(metal_hu_values)
        }

    # 分析金属周围的伪影区域
    if len(image.shape) == 3:
        # 扩展金属掩膜以分析周围区域
        dilated_mask = ndimage.binary_dilation(metal_mask,
                                              structure=np.ones((5,5,5)))
        surrounding_region = dilated_mask & (~metal_mask)

        if np.sum(surrounding_region) > 0:
            surrounding_hu = image[surrounding_region]
            analysis['surrounding_hu_stats'] = {
                'mean': np.mean(surrounding_hu),
                'std': np.std(surrounding_hu),
                'min': np.min(surrounding_hu),
                'max': np.max(surrounding_hu)
            }

    # 伪影严重程度评估
    analysis['severity'] = evaluate_artifact_severity(stats, analysis)

    return analysis

def evaluate_artifact_severity(stats, analysis):
    """
    评估金属伪影的严重程度

    参数:
        stats (dict): 检测统计信息
        analysis (dict): 金属伪影分析

    返回:
        str: 严重程度等级
    """
    metal_percentage = stats['metal_percentage']
    metal_count = stats['metal_count']

    if metal_count == 0:
        return "无伪影"
    elif metal_percentage > 5.0 or metal_count > 10:
        return "严重"
    elif metal_percentage > 1.0 or metal_count > 5:
        return "中等"
    elif metal_percentage > 0.1 or metal_count > 1:
        return "轻微"
    else:
        return "极轻微"

def generate_metal_phantom_data(shape=(256, 256, 100), metal_positions=None):
    """
    生成含有金属伪影的合成CT数据

    参数:
        shape (tuple): 图像形状
        metal_positions (list): 金属植入物位置列表

    返回:
        numpy.ndarray: 合成的CT图像
        list: 金属植入物信息
    """
    print("生成含金属伪影的合成CT数据...")

    # 创建基础CT数据
    z, y, x = np.meshgrid(np.linspace(0, 1, shape[0]),
                          np.linspace(0, 1, shape[1]),
                          np.linspace(0, 1, shape[2]), indexing='ij')

    # 背景软组织
    image = np.full(shape, 40.0)

    # 添加骨骼结构
    bone_center = [0.5, 0.5, 0.5]
    bone_a, bone_b, bone_c = 0.25, 0.3, 0.2
    bone_mask = ((x - bone_center[2])**2 / bone_a**2 +
                 (y - bone_center[1])**2 / bone_b**2 +
                 (z - bone_center[0])**2 / bone_c**2) <= 1
    image[bone_mask] = 800

    # 默认金属位置
    if metal_positions is None:
        metal_positions = [
            {'center': [0.3, 0.7, 0.4], 'radius': 0.03, 'hu_value': 4000, 'shape': 'sphere'},
            {'center': [0.6, 0.3, 0.6], 'radius': 0.02, 'hu_value': 3500, 'shape': 'sphere'},
            {'center': [0.7, 0.8, 0.2], 'size': [0.05, 0.03, 0.01], 'hu_value': 4500, 'shape': 'ellipsoid'}
        ]

    # 添加金属植入物
    metal_info = []
    for i, metal in enumerate(metal_positions):
        center = metal['center']
        hu_value = metal['hu_value']

        if metal['shape'] == 'sphere':
            radius = metal['radius']
            mask = ((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2) <= radius**2
        elif metal['shape'] == 'ellipsoid':
            size = metal['size']
            mask = ((x - center[2])**2 / size[2]**2 +
                   (y - center[1])**2 / size[1]**2 +
                   (z - center[0])**2 / size[0]**2) <= 1

        image[mask] = hu_value

        # 添加条状伪影
        add_streak_artifacts(image, mask, center, hu_value)

        metal_info.append({
            'id': i,
            'center': center,
            'shape': metal['shape'],
            'hu_value': hu_value,
            'mask': mask
        })

    # 添加噪声
    noise = np.random.normal(0, 20, shape)
    image += noise

    print(f"合成数据生成完成，包含 {len(metal_info)} 个金属植入物")
    return image, metal_info

def add_streak_artifacts(image, metal_mask, metal_center, metal_hu):
    """
    添加条状伪影（简化的伪影模型）

    参数:
        image (numpy.ndarray): 输入图像
        metal_mask (numpy.ndarray): 金属区域掩膜
        metal_center (list): 金属中心坐标
        metal_hu (float): 金属HU值
    """
    # 获取金属区域的中心坐标
    z_center, y_center, x_center = [
        int(metal_center[i] * image.shape[i]) for i in range(3)
    ]

    # 创建从金属中心向外辐射的条状伪影
    artifact_strength = (metal_hu - 1000) / 1000  # 归一化伪影强度

    # 在多个方向上创建伪影
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for radius in range(5, min(image.shape)//4, 2):
            # 计算伪影位置
            z_art = int(z_center + radius * np.cos(angle))
            y_art = int(y_center + radius * np.sin(angle))

            if 0 <= z_art < image.shape[0] and 0 <= y_art < image.shape[1]:
                # 创建径向伪影
                artifact_intensity = artifact_strength * 500 / (1 + radius/10)
                image[z_art, y_art, :] += artifact_intensity * np.random.randn(image.shape[2])

def visualize_metal_detection(image, metal_mask, stats, analysis, save_path=None):
    """
    可视化金属伪影检测结果

    参数:
        image (numpy.ndarray): 原始CT图像
        metal_mask (numpy.ndarray): 金属区域掩膜
        stats (dict): 检测统计信息
        analysis (dict): 金属伪影分析
        save_path (str): 保存路径
    """
    if len(image.shape) == 3:
        visualize_3d_metal_detection(image, metal_mask, stats, analysis, save_path)
    else:
        visualize_2d_metal_detection(image, metal_mask, stats, analysis, save_path)

def visualize_3d_metal_detection(image, metal_mask, stats, analysis, save_path=None):
    """
    可视化3D金属伪影检测结果
    """
    # 选择中间切片进行可视化
    mid_slice_z = image.shape[0] // 2
    mid_slice_y = image.shape[1] // 2
    mid_slice_x = image.shape[2] // 2

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 轴向切片 (Z方向)
    axes[0, 0].imshow(image[mid_slice_z], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 0].set_title(f'轴向切片 Z={mid_slice_z}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(metal_mask[mid_slice_z], cmap='Reds')
    axes[0, 1].set_title('金属掩膜 (轴向)')
    axes[0, 1].axis('off')

    # 矢状切片 (Y方向)
    axes[0, 2].imshow(image[:, mid_slice_y, :], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 2].set_title(f'矢状切片 Y={mid_slice_y}')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(metal_mask[:, mid_slice_y, :], cmap='Reds')
    axes[0, 3].set_title('金属掩膜 (矢状)')
    axes[0, 3].axis('off')

    # 冠状切片 (X方向)
    axes[1, 0].imshow(image[:, :, mid_slice_x], cmap='gray', vmin=-1000, vmax=1000)
    axes[1, 0].set_title(f'冠状切片 X={mid_slice_x}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(metal_mask[:, :, mid_slice_x], cmap='Reds')
    axes[1, 1].set_title('金属掩膜 (冠状)')
    axes[1, 1].axis('off')

    # HU值分布直方图
    axes[1, 2].hist(image.flatten(), bins=200, alpha=0.7, color='blue',
                   range=[-1000, 5000])
    axes[1, 2].axvline(stats['threshold'], color='red', linestyle='--',
                       label=f'阈值: {stats["threshold"]}')
    axes[1, 2].set_title('HU值分布')
    axes[1, 2].set_xlabel('HU值')
    axes[1, 2].set_ylabel('像素数量')
    axes[1, 2].legend()
    axes[1, 2].set_xlim([-1000, 5000])

    # 3D投影
    metal_projection = np.max(metal_mask, axis=0)
    axes[1, 3].imshow(metal_projection, cmap='Reds')
    axes[1, 3].set_title('金属区域3D投影')
    axes[1, 3].axis('off')

    # 添加统计信息
    fig.suptitle(f'金属伪影检测结果\n'
                f'检测到 {stats["metal_count"]} 个金属区域, '
                f'总计 {stats["total_metal_pixels"]:,} 像素 ({stats["metal_percentage"]:.2f}%)\n'
                f'伪影严重程度: {analysis["severity"]}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

def visualize_2d_metal_detection(image, metal_mask, stats, analysis, save_path=None):
    """
    可视化2D金属伪影检测结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始图像
    im1 = axes[0, 0].imshow(image, cmap='gray', vmin=-1000, vmax=4000)
    axes[0, 0].set_title('原始CT图像')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 金属掩膜
    axes[0, 1].imshow(metal_mask, cmap='Reds')
    axes[0, 1].set_title('检测到的金属区域')
    axes[0, 1].axis('off')

    # 叠加显示
    overlay = np.zeros((*image.shape, 3))
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    overlay[:, :, 0] = normalized_image  # Red channel
    overlay[:, :, 1] = normalized_image  # Green channel
    overlay[:, :, 2] = normalized_image  # Blue channel
    overlay[metal_mask, 0] = 1  # Highlight metal in red
    overlay[metal_mask, 1] = 0
    overlay[metal_mask, 2] = 0

    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('金属区域叠加显示')
    axes[0, 2].axis('off')

    # HU值分布
    axes[1, 0].hist(image.flatten(), bins=200, alpha=0.7, color='blue',
                   range=[-1000, 5000])
    axes[1, 0].axvline(stats['threshold'], color='red', linestyle='--',
                       label=f'阈值: {stats["threshold"]}')
    axes[1, 0].set_title('HU值分布')
    axes[1, 0].set_xlabel('HU值')
    axes[1, 0].set_ylabel('像素数量')
    axes[1, 0].legend()
    axes[1, 0].set_xlim([-1000, 5000])

    # 金属区域统计
    if stats['regions']:
        region_ids = [r['region_id'] for r in stats['regions']]
        region_areas = [r['area'] for r in stats['regions']]
        axes[1, 1].bar(region_ids, region_areas)
        axes[1, 1].set_title('各金属区域面积')
        axes[1, 1].set_xlabel('区域ID')
        axes[1, 1].set_ylabel('像素数量')

    # 严重程度评估
    severity_text = f"伪影严重程度: {analysis['severity']}\n"
    severity_text += f"金属区域数量: {stats['metal_count']}\n"
    severity_text += f"金属像素比例: {stats['metal_percentage']:.3f}%"

    # 输出检测结果摘要到控制台
    print("检测结果摘要:")
    print(severity_text)

    axes[1, 2].set_title('检测结果摘要')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

def demonstrate_different_thresholds(image):
    """
    演示不同阈值对检测结果的影响

    参数:
        image (numpy.ndarray): 输入CT图像
    """
    print("\n" + "="*60)
    print("不同阈值的金属检测结果对比")
    print("="*60)

    thresholds = [2000, 2500, 3000, 3500, 4000]
    results = {}

    for threshold in thresholds:
        print(f"\n测试阈值: {threshold} HU")
        metal_mask, stats = detect_metal_artifacts(image, threshold=threshold, min_area=50)
        analysis = analyze_metal_artifacts(image, metal_mask, stats)

        results[threshold] = {
            'metal_mask': metal_mask,
            'stats': stats,
            'analysis': analysis
        }

        print(f"  检测到 {stats['metal_count']} 个金属区域")
        print(f"  金属像素: {stats['total_metal_pixels']:,}")
        print(f"  严重程度: {analysis['severity']}")

    # 可视化对比
    fig, axes = plt.subplots(2, len(thresholds), figsize=(5*len(thresholds), 10))

    for i, threshold in enumerate(thresholds):
        result = results[threshold]
        mask = result['metal_mask']
        stats = result['stats']

        # 显示金属掩膜
        if len(image.shape) == 3:
            mask_2d = np.max(mask, axis=0)
        else:
            mask_2d = mask

        axes[0, i].imshow(mask_2d, cmap='Reds')
        axes[0, i].set_title(f'阈值 {threshold} HU\n{stats["metal_count"]} 个区域')
        axes[0, i].axis('off')

        # 显示统计信息
        info_text = f"像素数: {stats['total_metal_pixels']:,}\n"
        info_text += f"比例: {stats['metal_percentage']:.2f}%\n"
        info_text += f"严重程度: {result['analysis']['severity']}"

        # 输出统计信息到控制台
        print(f"阈值 {threshold} HU 统计信息:")
        print(info_text)

        axes[1, i].set_title(f'统计信息')
        axes[1, i].axis('off')

    plt.suptitle('不同阈值的金属检测结果对比', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    save_path = "output/metal_threshold_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.pause(2)  # 展示2秒
    plt.close()

def main():
    """
    主函数：演示金属伪影检测的完整流程
    """
    print("CT金属伪影检测演示")
    print("="*50)

    # 生成合成数据
    try:
        ct_image, metal_info = generate_metal_phantom_data(shape=(128, 128, 64))
        print(f"生成了包含 {len(metal_info)} 个金属植入物的测试数据")
    except Exception as e:
        print(f"数据生成失败: {e}")
        return

    # 金属伪影检测
    print("\n" + "="*50)
    print("金属伪影检测")
    print("="*50)

    metal_mask, stats = detect_metal_artifacts(
        ct_image,
        threshold=3000,
        min_area=100,
        morphological_operation=True
    )

    # 分析检测结果
    analysis = analyze_metal_artifacts(ct_image, metal_mask, stats)

    # 打印检测结果
    print(f"\n检测结果:")
    print(f"  检测到 {stats['metal_count']} 个金属区域")
    print(f"  金属像素总数: {stats['total_metal_pixels']:,}")
    print(f"  金属像素比例: {stats['metal_percentage']:.3f}%")
    print(f"  伪影严重程度: {analysis['severity']}")

    if 'metal_hu_stats' in analysis:
        metal_stats = analysis['metal_hu_stats']
        print(f"  金属HU值统计:")
        print(f"    均值: {metal_stats['mean']:.1f} HU")
        print(f"    范围: [{metal_stats['min']:.1f}, {metal_stats['max']:.1f}] HU")

    # 可视化结果
    os.makedirs("output", exist_ok=True)
    save_path = "output/metal_artifact_detection.png"
    visualize_metal_detection(ct_image, metal_mask, stats, analysis, save_path)

    # 演示不同阈值的影响
    demonstrate_different_thresholds(ct_image)

    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("1. 金属伪影检测是CT质量控制的重要步骤")
    print("2. 阈值选择影响检测敏感性和特异性")
    print("3. 连通性分析可以去除噪声干扰")
    print("4. 面积过滤可以保留有意义的金属植入物")
    print("5. 形态学操作可以优化检测结果")
    print("6. 伪影严重程度评估有助于后续处理决策")

if __name__ == "__main__":
    main()