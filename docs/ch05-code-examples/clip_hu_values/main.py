#!/usr/bin/env python3
"""
CT HU值截断处理
实现医学影像中CT图像的HU值截断和预处理

学习目标：
1. 理解CT HU值的物理意义和范围
2. 掌握HU值截断的方法和原理
3. 了解不同截断策略的临床应用

算法原理：
HU (Hounsfield Unit) 是CT图像的物理度量标准：
HU = 1000 × (μ_tissue - μ_water) / (μ_water - μ_air)

其中：
- μ_tissue: 组织的线性衰减系数
- μ_water: 水的线性衰减系数
- μ_air: 空气的线性衰减系数

临床意义：
- 空气: -1000 HU
- 水: 0 HU
- 软组织: -100 到 +100 HU
- 骨骼: +200 到 +3000+ HU
"""

import numpy as np
import matplotlib.pyplot as plt
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

def clip_hu_values(image, min_hu=-1000, max_hu=1000):
    """
    HU值截断：去除极端值，保留感兴趣组织范围

    参数:
        image (numpy.ndarray): 输入的CT图像数组
        min_hu (float): 最小HU值阈值
        max_hu (float): 最大HU值阈值

    返回:
        numpy.ndarray: 截断后的图像

    算法说明：
        1. 深拷贝避免修改原始数据
        2. 使用numpy布尔索引进行截断
        3. 保留[min_hu, max_hu]范围内的值
    """
    # 深拷贝避免修改原始数据
    processed_image = image.copy()

    # 截断HU值
    processed_image[processed_image < min_hu] = min_hu
    processed_image[processed_image > max_hu] = max_hu

    return processed_image

def analyze_hu_distribution(image, title="HU值分布"):
    """
    分析HU值的分布特征

    参数:
        image (numpy.ndarray): 输入图像
        title (str): 图表标题

    返回:
        dict: 统计信息
    """
    stats = {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'median': np.median(image),
        'q25': np.percentile(image, 25),
        'q75': np.percentile(image, 75)
    }

    print(f"\n{title}")
    print(f"均值: {stats['mean']:.2f} HU")
    print(f"标准差: {stats['std']:.2f} HU")
    print(f"最小值: {stats['min']:.2f} HU")
    print(f"最大值: {stats['max']:.2f} HU")
    print(f"中位数: {stats['median']:.2f} HU")
    print(f"25%分位数: {stats['q25']:.2f} HU")
    print(f"75%分位数: {stats['q75']:.2f} HU")

    return stats

def visualize_hu_clipping(original, clipped, min_hu, max_hu, save_path=None):
    """
    可视化HU值截断效果

    参数:
        original (numpy.ndarray): 原始图像
        clipped (numpy.ndarray): 截断后图像
        min_hu (float): 最小HU值
        max_hu (float): 最大HU值
        save_path (str): 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始图像中间切片
    mid_slice = original.shape[0] // 2
    axes[0, 0].imshow(original[mid_slice], cmap='gray', vmin=min_hu, vmax=max_hu)
    axes[0, 0].set_title('原始图像 (中间切片)')
    axes[0, 0].axis('off')

    # 截断后图像中间切片
    axes[0, 1].imshow(clipped[mid_slice], cmap='gray', vmin=min_hu, vmax=max_hu)
    axes[0, 1].set_title('截断后图像 (中间切片)')
    axes[0, 1].axis('off')

    # 差异图像
    diff = clipped - original
    axes[0, 2].imshow(diff[mid_slice], cmap='RdBu_r',
                     vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[0, 2].set_title('差异图像 (截断-原始)')
    axes[0, 2].axis('off')

    # 原始HU值分布直方图
    axes[1, 0].hist(original.flatten(), bins=200, alpha=0.7, color='blue',
                   range=[min_hu, max_hu])
    axes[1, 0].set_title('原始HU值分布')
    axes[1, 0].set_xlabel('HU值')
    axes[1, 0].set_ylabel('像素数量')
    axes[1, 0].axvline(min_hu, color='red', linestyle='--', alpha=0.7, label=f'最小阈值: {min_hu}')
    axes[1, 0].axvline(max_hu, color='red', linestyle='--', alpha=0.7, label=f'最大阈值: {max_hu}')
    axes[1, 0].legend()

    # 截断后HU值分布直方图
    axes[1, 1].hist(clipped.flatten(), bins=200, alpha=0.7, color='green',
                   range=[min_hu, max_hu])
    axes[1, 1].set_title('截断后HU值分布')
    axes[1, 1].set_xlabel('HU值')
    axes[1, 1].set_ylabel('像素数量')

    # 分布对比
    axes[1, 2].hist(original.flatten(), bins=200, alpha=0.5, color='blue',
                   label='原始', range=[min_hu, max_hu])
    axes[1, 2].hist(clipped.flatten(), bins=200, alpha=0.5, color='green',
                   label='截断后', range=[min_hu, max_hu])
    axes[1, 2].set_title('分布对比')
    axes[1, 2].set_xlabel('HU值')
    axes[1, 2].set_ylabel('像素数量')
    axes[1, 2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

def generate_synthetic_ct_data(shape=(256, 256, 128), noise_level=0.1):
    """
    生成合成的CT数据用于测试

    参数:
        shape (tuple): 图像形状
        noise_level (float): 噪声水平

    返回:
        numpy.ndarray: 合成的CT图像
    """
    print("生成合成CT数据用于测试...")

    # 创建3D坐标网格
    z, y, x = np.meshgrid(np.linspace(0, 1, shape[0]),
                          np.linspace(0, 1, shape[1]),
                          np.linspace(0, 1, shape[2]), indexing='ij')

    # 背景空气
    image = np.full(shape, -1000.0)

    # 添加椭圆形软组织区域
    center = [0.5, 0.5, 0.5]
    a, b, c = 0.3, 0.4, 0.35  # 椭球半轴

    mask = ((x - center[2])**2 / a**2 +
            (y - center[1])**2 / b**2 +
            (z - center[0])**2 / c**2) <= 1

    image[mask] = 40  # 软组织HU值

    # 添加骨骼结构
    bone_center = [0.5, 0.5, 0.3]
    bone_a, bone_b, bone_c = 0.15, 0.2, 0.1

    bone_mask = ((x - bone_center[2])**2 / bone_a**2 +
                 (y - bone_center[1])**2 / bone_b**2 +
                 (z - bone_center[0])**2 / bone_c**2) <= 1

    image[bone_mask] = 800  # 骨骼HU值

    # 添加金属植入物（极端HU值）
    metal_pos = [int(shape[0]*0.6), int(shape[1]*0.7), int(shape[2]*0.4)]
    metal_size = 5
    metal_mask = np.zeros(shape, dtype=bool)
    metal_mask[max(0,metal_pos[0]-metal_size):min(shape[0],metal_pos[0]+metal_size),
               max(0,metal_pos[1]-metal_size):min(shape[1],metal_pos[1]+metal_size),
               max(0,metal_pos[2]-metal_size):min(shape[2],metal_pos[2]+metal_size)] = True

    image[metal_mask] = 4000  # 金属植入物HU值

    # 添加高斯噪声
    noise = np.random.normal(0, noise_level * 50, shape)
    image += noise

    print(f"合成数据生成完成，形状: {shape}")
    return image

def demonstrate_clipping_strategies(image):
    """
    演示不同的HU值截断策略

    参数:
        image (numpy.ndarray): 输入CT图像
    """
    print("\n" + "="*60)
    print("HU值截断策略演示")
    print("="*60)

    # 常用截断策略
    strategies = [
        {"name": "全身范围", "min_hu": -1000, "max_hu": 1000, "desc": "包含大多数临床相关结构"},
        {"name": "软组织范围", "min_hu": -200, "max_hu": 400, "desc": "排除空气和致密骨"},
        {"name": "骨组织范围", "min_hu": -200, "max_hu": 3000, "desc": "适用于骨骼分析"},
        {"name": "肺窗范围", "min_hu": -1500, "max_hu": 600, "desc": "适用于肺部检查"},
    ]

    results = {}

    for strategy in strategies:
        print(f"\n{'-'*40}")
        print(f"策略: {strategy['name']}")
        print(f"范围: [{strategy['min_hu']}, {strategy['max_hu']}] HU")
        print(f"说明: {strategy['desc']}")

        # 应用截断
        clipped = clip_hu_values(image, strategy['min_hu'], strategy['max_hu'])

        # 分析截断前后
        original_stats = analyze_hu_distribution(image, f"原始图像统计")
        clipped_stats = analyze_hu_distribution(clipped, f"截断后统计")

        # 计算截断的像素数量
        clipped_pixels = np.sum((image < strategy['min_hu']) | (image > strategy['max_hu']))
        total_pixels = image.size
        clip_percentage = (clipped_pixels / total_pixels) * 100

        print(f"截断像素数量: {clipped_pixels:,}")
        print(f"截断像素比例: {clip_percentage:.2f}%")

        results[strategy['name']] = {
            'clipped_image': clipped,
            'original_stats': original_stats,
            'clipped_stats': clipped_stats,
            'clip_percentage': clip_percentage
        }

        # 可视化效果
        os.makedirs("output", exist_ok=True)
        save_path = f"output/hu_clipping_{strategy['name'].replace(' ', '_')}.png"
        visualize_hu_clipping(image, clipped, strategy['min_hu'], strategy['max_hu'], save_path)

    return results

def main():
    """
    主函数：演示HU值截断的完整流程
    """
    print("CT HU值截断处理演示")
    print("="*50)

    # 生成或加载测试数据
    try:
        # 尝试生成合成数据
        ct_image = generate_synthetic_ct_data(shape=(128, 128, 64), noise_level=0.1)
    except Exception as e:
        print(f"数据生成失败: {e}")
        return

    # 演示不同截断策略
    results = demonstrate_clipping_strategies(ct_image)

    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("1. HU值截断是CT预处理的重要步骤")
    print("2. 不同临床应用需要不同的截断策略")
    print("3. 软组织范围适用于内脏器官分析")
    print("4. 骨组织范围适用于骨骼相关应用")
    print("5. 肺窗范围适用于肺部疾病检查")
    print("6. 截断会丢失部分信息，但能提高后续处理的稳定性")

if __name__ == "__main__":
    main()