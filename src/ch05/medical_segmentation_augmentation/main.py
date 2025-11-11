#!/usr/bin/env python3
"""
医学图像分割专门数据增强效果演示 / Medical Image Segmentation Augmentation Demo
功能：展示解剖学约束的医学图像增强技术
Enhanced Features: Elastic deformation, intensity transformation, noise addition, partial occlusion
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize, rotate, AffineTransform, warp
from skimage.filters import gaussian
import os
from pathlib import Path
import json
import warnings

# 过滤matplotlib警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*findfont.*")
warnings.filterwarnings("ignore", message=".*font.*")

# 设置中文字体 / Set Chinese font
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
    except:
        pass

class MedicalSegmentationAugmentation:
    """
    医学图像分割的专门数据增强类
    Medical Image Segmentation Specific Augmentation Class
    """

    def __init__(self, seed=42):
        """
        初始化增强参数
        Initialize augmentation parameters
        """
        np.random.seed(seed)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def create_sample_medical_image(self):
        """
        创建示例医学图像（模拟CT肺野）
        Create sample medical image (simulated CT lung field)
        """
        print("创建示例医学图像 / Creating sample medical image...")

        # 创建512x512的图像
        image = np.zeros((512, 512), dtype=np.float32)

        # 模拟肺野区域（椭圆形状）
        # 左肺
        center_left = (200, 256)
        axes_left = (120, 160)
        angle_left = 30

        # 右肺
        center_right = (312, 256)
        axes_right = (120, 160)
        angle_right = -30

        # 创建肺野掩码
        y, x = np.ogrid[:512, :512]

        # 左肺椭圆
        cos_left = np.cos(np.radians(angle_left))
        sin_left = np.sin(np.radians(angle_left))
        x_left = (x - center_left[0]) * cos_left + (y - center_left[1]) * sin_left
        y_left = -(x - center_left[0]) * sin_left + (y - center_left[1]) * cos_left
        left_lung = ((x_left/axes_left[0])**2 + (y_left/axes_left[1])**2) <= 1

        # 右肺椭圆
        cos_right = np.cos(np.radians(angle_right))
        sin_right = np.sin(np.radians(angle_right))
        x_right = (x - center_right[0]) * cos_right + (y - center_right[1]) * sin_right
        y_right = -(x - center_right[0]) * sin_right + (y - center_right[1]) * cos_right
        right_lung = ((x_right/axes_right[0])**2 + (y_right/axes_right[1])**2) <= 1

        # 肺野掩码
        lung_mask = (left_lung | right_lung).astype(np.float32)

        # 添加肺部纹理（模拟肺血管和支气管）
        noise = np.random.randn(512, 512) * 0.1
        texture = gaussian(noise, sigma=2)

        # 肺部密度（HU值模拟）
        lung_density = -800 + texture * 100  # 肺部HU值约-800
        body_density = 0  # 软组织HU值约0

        # 组合图像
        image = np.where(lung_mask > 0.5, lung_density, body_density)

        # 添加小病灶（模拟结节）
        nodule_center = (250, 200)
        nodule_radius = 15
        nodule_mask = ((x - nodule_center[0])**2 + (y - nodule_center[1])**2) <= nodule_radius**2
        nodule_density = -300 + np.random.randn() * 50  # 结节密度
        image = np.where(nodule_mask, nodule_density, image)

        # 归一化到0-255显示
        image_display = ((image + 1000) / 1000 * 255).clip(0, 255).astype(np.uint8)
        mask_display = (lung_mask * 255).astype(np.uint8)

        print(f"  图像尺寸 Image size: {image.shape}")
        print(f"  肺野占比 Lung ratio: {np.mean(lung_mask):.2%}")
        print(f"  密度范围 Density range: [{image.min():.1f}, {image.max():.1f}] HU")

        return image, image_display, lung_mask, mask_display, nodule_mask

    def elastic_deformation(self, image, mask, alpha=1000, sigma=8):
        """
        弹性变形：模拟呼吸、心脏运动等生理变化
        Elastic deformation: Simulate physiological changes like breathing, cardiac motion

        参数 Parameters:
            image: 输入图像 Input image
            mask: 分割掩码 Segmentation mask
            alpha: 变形强度 Deformation strength
            sigma: 平滑程度 Smoothness level
        """
        print(f"执行弹性变形 / Applying elastic deformation (α={alpha}, σ={sigma})")

        # 生成随机位移场
        shape = image.shape
        dx = gaussian(np.random.randn(*shape), sigma, mode='reflect') * alpha
        dy = gaussian(np.random.randn(*shape), sigma, mode='reflect') * alpha

        # 创建网格坐标
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1], dtype=np.float32))

        # 应用位移
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # 使用scipy进行变形
        warped_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
        warped_mask = ndimage.map_coordinates(mask, indices, order=0, mode='reflect')
        warped_image = warped_image.reshape(image.shape)
        warped_mask = warped_mask.reshape(mask.shape)

        return warped_image, warped_mask

    def intensity_transform(self, image, mask, contrast_factor=1.2, brightness_shift=0):
        """
        强度变换：模拟不同扫描参数和设备差异
        Intensity transformation: Simulate different scanning parameters and device differences

        参数 Parameters:
            image: 输入图像 Input image
            mask: 分割掩码 Segmentation mask
            contrast_factor: 对比度因子 Contrast factor
            brightness_shift: 亮度偏移 Brightness shift
        """
        print(f"执行强度变换 / Applying intensity transformation (contrast={contrast_factor}, brightness={brightness_shift})")

        # 应用对比度和亮度调整
        transformed = image * contrast_factor + brightness_shift

        # 保持HU值范围合理
        transformed = np.clip(transformed, -1000, 1000)

        return transformed, mask

    def add_noise(self, image, mask, noise_type='gaussian', noise_level=20):
        """
        噪声添加：模拟真实临床环境的图像噪声
        Noise addition: Simulate real clinical environment image noise

        参数 Parameters:
            image: 输入图像 Input image
            mask: 分割掩码 Segmentation mask
            noise_type: 噪声类型 Noise type ('gaussian', 'poisson', 'speckle')
            noise_level: 噪声强度 Noise level
        """
        print(f"添加{noise_type}噪声 / Adding {noise_type} noise (level={noise_level})")

        if noise_type == 'gaussian':
            # 高斯噪声（模拟电子噪声）
            noise = np.random.normal(0, noise_level, image.shape)
            noisy_image = image + noise

        elif noise_type == 'poisson':
            # 泊松噪声（模拟量子噪声）
            # 先将图像缩放到正值范围
            scaled = (image - image.min()) / (image.max() - image.min()) * 100
            noisy_image = np.random.poisson(scaled * noise_level / 10) / (noise_level / 10)
            noisy_image = noisy_image / np.max(scaled) * (image.max() - image.min()) + image.min()

        elif noise_type == 'speckle':
            # 斑点噪声（模拟超声噪声）
            noise = np.random.randn(*image.shape)
            noisy_image = image + image * noise * (noise_level / 100)

        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        return noisy_image, mask

    def add_partial_occlusion(self, image, mask, occlusion_type='metal', severity=0.3):
        """
        部分遮挡：模拟金属伪影、运动伪影等
        Partial occlusion: Simulate metal artifacts, motion artifacts, etc.

        参数 Parameters:
            image: 输入图像 Input image
            mask: 分割掩码 Segmentation mask
            occlusion_type: 遮挡类型 Occlusion type ('metal', 'motion', 'grid')
            severity: 遮挡严重程度 Occlusion severity (0-1)
        """
        print(f"添加{occlusion_type}遮挡 / Adding {occlusion_type} occlusion (severity={severity})")

        occluded_image = image.copy()
        occluded_mask = mask.copy()

        if occlusion_type == 'metal':
            # 金属伪影（线性条纹）
            n_lines = int(5 * severity)
            for _ in range(n_lines):
                start_x = np.random.randint(0, image.shape[1])
                start_y = 0
                end_x = start_x + np.random.randint(-20, 20)
                end_y = image.shape[0]

                # 创建线条遮罩
                y_coords = np.arange(start_y, end_y)
                x_coords = np.linspace(start_x, end_x, len(y_coords)).astype(int)
                x_coords = np.clip(x_coords, 0, image.shape[1]-1)

                # 添加金属伪影（高密度条纹）
                for y, x in zip(y_coords, x_coords):
                    if 0 <= y < image.shape[0]:
                        # 条纹宽度
                        for dx in range(-2, 3):
                            if 0 <= x+dx < image.shape[1]:
                                occluded_image[y, x+dx] = 2000  # 金属密度HU值

        elif occlusion_type == 'motion':
            # 运动伪影（模糊方向）
            angle = np.random.uniform(0, 2*np.pi)
            distance = int(20 * severity)

            # 创建运动模糊核
            kernel_size = distance * 2 + 1
            kernel = np.zeros((kernel_size, kernel_size))

            center = kernel_size // 2
            for i in range(kernel_size):
                x = int(center + (i - center) * np.cos(angle))
                y = int(center + (i - center) * np.sin(angle))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1

            # 归一化核
            kernel = kernel / np.sum(kernel)

            # 应用运动模糊（使用scipy）
            from scipy.signal import convolve2d
            occluded_image = convolve2d(occluded_image, kernel, mode='same', boundary='symm')

        elif occlusion_type == 'grid':
            # 网格伪影（模拟探测器失效）
            grid_spacing = max(10, int(50 * (1 - severity)))

            # 创建网格
            grid = np.zeros_like(image)
            grid[::grid_spacing, :] = 1
            grid[:, ::grid_spacing] = 1

            # 应用网格遮挡
            occluded_image = np.where(grid > 0, 0, occluded_image)

        return occluded_image, occluded_mask

    def visualize_augmentation_results(self, original_img, original_mask, augmentations):
        """
        可视化增强结果
        Visualize augmentation results
        """
        print("生成增强效果可视化 / Generating augmentation visualization...")

        # 计算显示范围
        all_images = [original_img] + [aug[0] for aug in augmentations]
        vmin = min(img.min() for img in all_images)
        vmax = max(img.max() for img in all_images)

        # 创建8面板布局
        fig = plt.figure(figsize=(20, 12))

        # 原始图像
        ax1 = plt.subplot(2, 4, 1)
        im1 = ax1.imshow(original_img, cmap='gray', vmin=vmin, vmax=vmax)
        ax1.set_title('原始图像 Original Image\n(模拟CT肺野 Simulated CT Lung)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='HU值 HU Value')

        # 原始掩码
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(original_mask, cmap='Reds')
        ax2.set_title('肺野掩码 Lung Mask\n(分割真值 Ground Truth)', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 叠加显示
        ax3 = plt.subplot(2, 4, 3)
        ax3.imshow(original_img, cmap='gray', vmin=vmin, vmax=vmax)
        ax3.imshow(original_mask, cmap='Reds', alpha=0.3)
        ax3.set_title('图像+掩码叠加 Image + Mask Overlay\n(病灶位置 Nodule Location)', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 统计信息
        ax4 = plt.subplot(2, 4, 4)
        ax4.axis('off')
        stats_text = f"""原始图像统计 Original Image Statistics:

尺寸 Size: {original_img.shape}
最小值 Min: {original_img.min():.1f} HU
最大值 Max: {original_img.max():.1f} HU
均值 Mean: {original_img.mean():.1f} HU
标准差 Std: {original_img.std():.1f} HU

肺野占比 Lung Ratio: {original_mask.mean():.2%}
病灶大小 Nodule Size: {np.sum(original_mask>0):.0f} pixels"""

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_title('图像信息 Image Information', fontsize=12, fontweight='bold')

        # 增强结果
        augmentation_names = [
            "弹性变形 Elastic Deformation\n(模拟呼吸运动 Simulate Breathing)",
            "强度变换 Intensity Transform\n(对比度调整 Contrast Adjustment)",
            "噪声添加 Noise Addition\n(高斯噪声 Gaussian Noise)",
            "部分遮挡 Partial Occlusion\n(金属伪影 Metal Artifacts)"
        ]

        for i, (augmented_img, augmented_mask, aug_name) in enumerate(augmentations):
            ax = plt.subplot(2, 4, i+5)

            # 显示增强后的图像
            im = ax.imshow(augmented_img, cmap='gray', vmin=vmin, vmax=vmax)
            # 叠加掩码
            ax.imshow(augmented_mask, cmap='Reds', alpha=0.3)

            ax.set_title(augmentation_names[i], fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        # 保存图像
        output_path = self.output_dir / "medical_segmentation_augmentation_demo.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存 / Visualization saved to: {output_path}")

        plt.close()

        return output_path

    def analyze_augmentation_effects(self, original_img, original_mask, augmentations):
        """
        分析增强效果
        Analyze augmentation effects
        """
        print("\n分析增强效果 / Analyzing augmentation effects...")

        analysis_results = {
            'original': {
                'mean': float(original_img.mean()),
                'std': float(original_img.std()),
                'min': float(original_img.min()),
                'max': float(original_img.max()),
                'lung_ratio': float(original_mask.mean())
            },
            'augmentations': {}
        }

        aug_types = ['elastic_deformation', 'intensity_transform', 'noise_addition', 'partial_occlusion']

        for i, (augmented_img, augmented_mask, _) in enumerate(augmentations):
            aug_type = aug_types[i]

            # 计算指标
            mse = np.mean((original_img - augmented_img) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

            # 掩码变化
            mask_diff = np.mean(np.abs(original_mask - augmented_mask))

            analysis_results['augmentations'][aug_type] = {
                'mean': float(augmented_img.mean()),
                'std': float(augmented_img.std()),
                'min': float(augmented_img.min()),
                'max': float(augmented_img.max()),
                'mse': float(mse),
                'psnr': float(psnr),
                'mask_change': float(mask_diff),
                'lung_ratio': float(augmented_mask.mean())
            }

        # 保存分析结果
        analysis_path = self.output_dir / "augmentation_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        print(f"分析结果已保存 / Analysis saved to: {analysis_path}")

        # 打印关键结果
        print("\n[Analysis] 增强效果分析 / Augmentation Effect Analysis:")
        print("-" * 60)

        for aug_type, metrics in analysis_results['augmentations'].items():
            aug_name = {
                'elastic_deformation': '弹性变形',
                'intensity_transform': '强度变换',
                'noise_addition': '噪声添加',
                'partial_occlusion': '部分遮挡'
            }.get(aug_type, aug_type)

            print(f"\n{aug_name} / {aug_name.replace(' ', ' ').title()}:")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  密度变化 Density change: {metrics['mean'] - analysis_results['original']['mean']:+.1f} HU")
            print(f"  掩码变化 Mask change: {metrics['mask_change']:.4f}")
            print(f"  标准差变化 Std change: {metrics['std'] - analysis_results['original']['std']:+.1f}")

        return analysis_results

def main():
    """
    主函数 / Main function
    """
    print("=" * 80)
    print("医学图像分割专门数据增强效果演示 / Medical Image Segmentation Augmentation Demo")
    print("=" * 80)

    # 初始化增强器
    augmentor = MedicalSegmentationAugmentation(seed=42)

    # 创建示例医学图像
    print("\n[Medical] 创建示例医学图像 / Creating sample medical image...")
    original_img, original_img_display, original_mask, original_mask_display, nodule_mask = augmentor.create_sample_medical_image()

    # 应用不同的增强技术
    print("\n[Augment] 应用增强技术 / Applying augmentation techniques...")
    augmentations = []

    # 1. 弹性变形
    elastic_img, elastic_mask = augmentor.elastic_deformation(
        original_img, original_mask, alpha=800, sigma=6
    )
    augmentations.append((elastic_img, elastic_mask, "弹性变形"))

    # 2. 强度变换
    intensity_img, intensity_mask = augmentor.intensity_transform(
        original_img, original_mask, contrast_factor=1.3, brightness_shift=50
    )
    augmentations.append((intensity_img, intensity_mask, "强度变换"))

    # 3. 噪声添加
    noise_img, noise_mask = augmentor.add_noise(
        original_img, original_mask, noise_type='gaussian', noise_level=15
    )
    augmentations.append((noise_img, noise_mask, "噪声添加"))

    # 4. 部分遮挡
    occlusion_img, occlusion_mask = augmentor.add_partial_occlusion(
        original_img, original_mask, occlusion_type='metal', severity=0.4
    )
    augmentations.append((occlusion_img, occlusion_mask, "部分遮挡"))

    # 可视化结果
    print("\n[Visualize] 生成可视化结果 / Generating visualization results...")
    viz_path = augmentor.visualize_augmentation_results(
        original_img_display, original_mask_display, augmentations
    )

    # 分析效果
    analysis_results = augmentor.analyze_augmentation_effects(
        original_img, original_mask, augmentations
    )

    print("\n" + "=" * 80)
    print("[Complete] 医学图像分割增强演示完成 / Medical Segmentation Augmentation Demo Completed!")
    print("=" * 80)

    return {
        'visualization_path': str(viz_path),
        'analysis_results': analysis_results
    }

if __name__ == "__main__":
    results = main()