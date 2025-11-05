#!/usr/bin/env python3
"""
医学图像重采样完整实现

功能：将不同分辨率、不同空间方向的医学影像重采样到统一的标准
算法：多种插值方法（线性、最近邻、三次样条、B样条）
应用：CT、MRI等多模态医学影像的分辨率标准化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResamplingConfig:
    """重采样配置参数"""
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # 目标间距 (mm)
    interpolation_method: str = 'linear'  # 插值方法
    origin_indexing: str = 'center'  # 原点索引方式
    preserve_intensity: bool = True  # 保持强度信息
    anti_aliasing: bool = True  # 抗锯齿
    prefilter: bool = True  # 预滤波

class MedicalImageResampler:
    """
    医学图像重采样器

    支持多种插值方法和配置选项的医学图像重采样工具
    """

    def __init__(self, config: ResamplingConfig):
        self.config = config
        self.interpolation_orders = {
            'nearest': 0,
            'linear': 1,
            'quadratic': 2,
            'cubic': 3,
            'quartic': 4,
            'quintic': 5
        }

    def calculate_new_shape(self, original_shape: Tuple[int, ...],
                           original_spacing: Tuple[float, ...],
                           target_spacing: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        计算重采样后的图像形状

        Args:
            original_shape: 原始图像形状
            original_spacing: 原始间距
            target_spacing: 目标间距

        Returns:
            新的图像形状
        """
        scale_factors = np.array(original_spacing) / np.array(target_spacing)
        new_shape = np.round(np.array(original_shape) * scale_factors).astype(int)
        return tuple(new_shape)

    def calculate_scale_factors(self, original_spacing: Tuple[float, ...],
                              target_spacing: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        计算缩放因子

        Args:
            original_spacing: 原始间距
            target_spacing: 目标间距

        Returns:
            缩放因子
        """
        return tuple(orig / target for orig, target in zip(original_spacing, target_spacing))

    def resample_image(self, image: np.ndarray,
                      original_spacing: Tuple[float, ...],
                      target_spacing: Optional[Tuple[float, ...]] = None,
                      method: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        重采样医学图像

        Args:
            image: 输入图像
            original_spacing: 原始间距
            target_spacing: 目标间距（可选，使用配置默认值）
            method: 插值方法（可选，使用配置默认值）

        Returns:
            重采样后的图像和重采样信息
        """
        # 使用默认配置
        if target_spacing is None:
            target_spacing = self.config.target_spacing
        if method is None:
            method = self.config.interpolation_method

        # 计算缩放因子和新形状
        scale_factors = self.calculate_scale_factors(original_spacing, target_spacing)
        new_shape = self.calculate_new_shape(image.shape, original_spacing, target_spacing)

        # 记录原始信息
        original_info = {
            'shape': image.shape,
            'spacing': original_spacing,
            'dtype': str(image.dtype),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image))
        }

        logger.info(f"开始重采样: {image.shape} -> {new_shape}")
        logger.info(f"缩放因子: {scale_factors}")

        # 选择插值方法
        if method in self.interpolation_orders:
            # scipy插值
            order = self.interpolation_orders[method]

            # 抗锯齿处理
            if self.config.anti_aliasing and order > 1:
                # 下采样时需要抗锯齿
                downscale_factors = [s for s in scale_factors if s < 1]
                if downscale_factors:
                    from scipy.ndimage import gaussian_filter
                    sigma = np.minimum([0.5 / s if s < 1 else 0.0 for s in scale_factors], 2.0)
                    image = gaussian_filter(image, sigma=tuple(sigma))

            # 执行重采样
            resampled_image = ndimage.zoom(
                image,
                scale_factors,
                order=order,
                mode='nearest',  # 边界处理
                prefilter=self.config.prefilter,
                output=image.dtype if self.config.preserve_intensity else np.float32
            )

        elif method == 'bspline':
            # B样条插值（需要SimpleITK）
            try:
                import SimpleITK as sitk

                # 转换为SimpleITK格式
                sitk_image = sitk.GetImageFromArray(image)
                sitk_image.SetSpacing(original_spacing)

                # 设置重采样器
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(target_spacing)
                resampler.SetSize(new_shape)
                resampler.SetOutputOrigin(sitk_image.GetOrigin())
                resampler.SetOutputDirection(sitk_image.GetDirection())

                # 选择插值器
                interpolators = {
                    'nearest': sitk.sitkNearestNeighbor,
                    'linear': sitk.sitkLinear,
                    'bspline': sitk.sitkBSpline,
                    'gaussian': sitk.sitkGaussian,
                    'lanczos': sitk.sitkLanczosWindowedSinc
                }
                resampler.SetInterpolator(interpolators.get(method, sitk.sitkLinear))

                # 执行重采样
                resampled = resampler.Execute(sitk_image)
                resampled_image = sitk.GetArrayFromImage(resampled)

            except ImportError:
                logger.warning("SimpleITK未安装，使用三次样条插值代替")
                resampled_image = ndimage.zoom(image, scale_factors, order=3)

        else:
            raise ValueError(f"不支持的插值方法: {method}")

        # 记录重采样信息
        resampling_info = {
            'original_info': original_info,
            'resampled_info': {
                'shape': resampled_image.shape,
                'spacing': target_spacing,
                'dtype': str(resampled_image.dtype),
                'min_value': float(np.min(resampled_image)),
                'max_value': float(np.max(resampled_image)),
                'mean_value': float(np.mean(resampled_image)),
                'std_value': float(np.std(resampled_image))
            },
            'parameters': {
                'scale_factors': scale_factors,
                'interpolation_method': method,
                'target_spacing': target_spacing,
                'anti_aliasing': self.config.anti_aliasing,
                'preserve_intensity': self.config.preserve_intensity
            }
        }

        logger.info(f"重采样完成: {resampled_image.shape}")
        return resampled_image, resampling_info

    def resample_with_landmarks(self, image: np.ndarray,
                               landmarks: np.ndarray,
                               original_spacing: Tuple[float, ...],
                               target_spacing: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """
        带关键点的重采样

        Args:
            image: 输入图像
            landmarks: 关键点坐标 (N, D)
            original_spacing: 原始间距
            target_spacing: 目标间距

        Returns:
            重采样后的图像和变换后的关键点
        """
        # 重采样图像
        resampled_image, _ = self.resample_image(image, original_spacing, target_spacing)

        # 变换关键点坐标
        scale_factors = self.calculate_scale_factors(original_spacing, target_spacing)
        resampled_landmarks = landmarks * np.array(scale_factors)

        return resampled_image, resampled_landmarks

def generate_synthetic_medical_image(shape=(128, 128, 64), modality='ct'):
    """
    生成合成医学图像

    Args:
        shape: 图像形状
        modality: 模态类型 ('ct', 'mri', 'pet')

    Returns:
        合成图像和间距信息
    """
    if modality == 'ct':
        # CT图像特征
        spacing = (0.7, 0.7, 2.5)  # 典型CT间距
        image = np.zeros(shape, dtype=np.int16)

        # 添加解剖结构
        center_z, center_y, center_x = shape[0]//2, shape[1]//2, shape[2]//2

        # 外轮廓（软组织）
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    dist = np.sqrt(((z-center_z)/(shape[0]*0.4))**2 +
                                 ((y-center_y)/(shape[1]*0.35))**2 +
                                 ((x-center_x)/(shape[2]*0.3))**2)
                    if dist < 1.0:
                        image[z, y, x] = -400 + 200 * dist  # 软组织HU值

        # 内部器官（心脏、肝脏等）
        organ_positions = [
            (center_z, center_y-20, center_x-10, 0.15, 60),   # 心脏
            (center_z+10, center_y+15, center_x+5, 0.12, 80), # 肝脏
            (center_z-5, center_y+10, center_x-15, 0.08, 40)  # 肾脏
        ]

        for z0, y0, x0, size_ratio, intensity in organ_positions:
            for z in range(shape[0]):
                for y in range(shape[1]):
                    for x in range(shape[2]):
                        dist = np.sqrt(((z-z0)/(shape[0]*size_ratio))**2 +
                                     ((y-y0)/(shape[1]*size_ratio))**2 +
                                     ((x-x0)/(shape[2]*size_ratio))**2)
                        if dist < 1.0:
                            image[z, y, x] = intensity

        # 添加噪声
        noise = np.random.normal(0, 15, shape)
        image = image + noise

    elif modality == 'mri':
        # MRI图像特征
        spacing = (1.0, 1.0, 3.0)  # 典型MRI间距
        image = np.zeros(shape, dtype=np.float32)

        # 脑部结构
        center_z, center_y, center_x = shape[0]//2, shape[1]//2, shape[2]//2

        # 白质和灰质
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    dist = np.sqrt(((z-center_z)/(shape[0]*0.4))**2 +
                                 ((y-center_y)/(shape[1]*0.35))**2 +
                                 ((x-center_x)/(shape[2]*0.3))**2)
                    if dist < 1.0:
                        # 白质（内部）
                        if dist < 0.6:
                            image[z, y, x] = 0.7 + 0.2 * np.random.random()
                        # 灰质（外部）
                        else:
                            image[z, y, x] = 0.4 + 0.2 * np.random.random()

        # 添加脑脊液
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    dist = np.sqrt(((z-center_z)/(shape[0]*0.5))**2 +
                                 ((y-center_y)/(shape[1]*0.45))**2 +
                                 ((x-center_x)/(shape[2]*0.4))**2)
                    if 0.9 < dist < 1.0:
                        image[z, y, x] = 0.1 + 0.05 * np.random.random()

    else:  # PET
        spacing = (2.0, 2.0, 3.0)  # 典型PET间距
        image = np.zeros(shape, dtype=np.float32)

        # 代谢活跃区域
        center_z, center_y, center_x = shape[0]//2, shape[1]//2, shape[2]//2

        # 背景代谢
        image[:] = 0.5 + 0.1 * np.random.random(shape)

        # 高代谢区域（肿瘤或炎症）
        hotspots = [
            (center_z-10, center_y-5, center_x+8, 0.08, 3.5),
            (center_z+5, center_y+10, center_x-10, 0.06, 2.8),
            (center_z, center_y, center_x, 0.1, 4.2)
        ]

        for z0, y0, x0, size_ratio, intensity in hotspots:
            for z in range(shape[0]):
                for y in range(shape[1]):
                    for x in range(shape[2]):
                        dist = np.sqrt(((z-z0)/(shape[0]*size_ratio))**2 +
                                     ((y-y0)/(shape[1]*size_ratio))**2 +
                                     ((x-x0)/(shape[2]*size_ratio))**2)
                        if dist < 1.0:
                            image[z, y, x] = max(image[z, y, x],
                                               intensity * np.exp(-dist**2))

    return image, spacing

def evaluate_resampling_quality(original_image: np.ndarray,
                              resampled_image: np.ndarray,
                              original_spacing: Tuple[float, ...],
                              resampled_spacing: Tuple[float, ...]) -> Dict:
    """
    评估重采样质量

    Args:
        original_image: 原始图像
        resampled_image: 重采样图像
        original_spacing: 原始间距
        resampled_spacing: 重采样间距

    Returns:
        质量评估指标
    """
    # 计算体素大小
    original_voxel_size = np.prod(original_spacing)
    resampled_voxel_size = np.prod(resampled_spacing)

    # 强度统计
    original_intensity = {
        'mean': float(np.mean(original_image)),
        'std': float(np.std(original_image)),
        'min': float(np.min(original_image)),
        'max': float(np.max(original_image))
    }

    resampled_intensity = {
        'mean': float(np.mean(resampled_image)),
        'std': float(np.std(resampled_image)),
        'min': float(np.min(resampled_image)),
        'max': float(np.max(resampled_image))
    }

    # 计算强度保持度
    intensity_correlation = np.corrcoef(
        original_image.flatten(),
        _resize_to_match(resampled_image, original_image.shape).flatten()
    )[0, 1]

    # 空间分辨率变化
    resolution_change = np.array(original_spacing) / np.array(resampled_spacing)

    # 计算信噪比（简化版本）
    signal_power = np.var(resampled_image)
    noise_power = np.var(np.diff(resampled_image.flatten()))
    snr = signal_power / (noise_power + 1e-8)

    return {
        'intensity_preservation': {
            'correlation': float(intensity_correlation),
            'original_stats': original_intensity,
            'resampled_stats': resampled_intensity,
            'mean_change_percent': float(abs(resampled_intensity['mean'] - original_intensity['mean']) / (abs(original_intensity['mean']) + 1e-8) * 100)
        },
        'spatial_resolution': {
            'voxel_size_change': float(resampled_voxel_size / original_voxel_size),
            'resolution_change_factors': resolution_change.tolist(),
            'isotropic_degree': float(np.std(resampled_spacing) / np.mean(resampled_spacing))
        },
        'image_quality': {
            'snr': float(snr),
            'dynamic_range': float(np.max(resampled_image) - np.min(resampled_image)),
            'contrast': float(np.std(resampled_image) / (np.mean(resampled_image) + 1e-8))
        }
    }

def _resize_to_match(image: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    将图像调整到目标形状（用于相关性计算）
    """
    if image.shape == target_shape:
        return image

    scale_factors = np.array(target_shape) / np.array(image.shape)
    return ndimage.zoom(image, scale_factors, order=1)

def visualize_resampling_results(original_image: np.ndarray,
                                resampled_image: np.ndarray,
                                original_spacing: Tuple[float, ...],
                                resampled_spacing: Tuple[float, ...],
                                quality_metrics: Dict,
                                save_path: Optional[str] = None):
    """
    可视化重采样结果

    Args:
        original_image: 原始图像
        resampled_image: 重采样图像
        original_spacing: 原始间距
        resampled_spacing: 重采样间距
        quality_metrics: 质量评估指标
        save_path: 保存路径
    """
    # 选择中间切片进行显示
    if len(original_image.shape) == 3:
        # 3D图像
        z_mid = original_image.shape[0] // 2
        y_mid = original_image.shape[1] // 2
        x_mid = original_image.shape[2] // 2

        z_mid_resampled = resampled_image.shape[0] // 2
        y_mid_resampled = resampled_image.shape[1] // 2
        x_mid_resampled = resampled_image.shape[2] // 2

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始图像
        im1 = axes[0, 0].imshow(original_image[z_mid, :, :], cmap='gray')
        axes[0, 0].set_title(f'原始图像 - Z切片\n间距: {original_spacing}')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        im2 = axes[0, 1].imshow(original_image[:, y_mid, :], cmap='gray')
        axes[0, 1].set_title(f'原始图像 - Y切片')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        im3 = axes[0, 2].imshow(original_image[:, :, x_mid], cmap='gray')
        axes[0, 2].set_title(f'原始图像 - X切片')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

        # 重采样图像
        im4 = axes[1, 0].imshow(resampled_image[z_mid_resampled, :, :], cmap='gray')
        axes[1, 0].set_title(f'重采样图像 - Z切片\n间距: {resampled_spacing}')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

        im5 = axes[1, 1].imshow(resampled_image[:, y_mid_resampled, :], cmap='gray')
        axes[1, 1].set_title(f'重采样图像 - Y切片')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

        im6 = axes[1, 2].imshow(resampled_image[:, :, x_mid_resampled], cmap='gray')
        axes[1, 2].set_title(f'重采样图像 - X切片')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    else:
        # 2D图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(f'原始图像\n间距: {original_spacing}')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        im2 = axes[1].imshow(resampled_image, cmap='gray')
        axes[1].set_title(f'重采样图像\n间距: {resampled_spacing}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # 差异图
        if original_image.shape == resampled_image.shape:
            diff_image = np.abs(original_image - resampled_image)
            im3 = axes[2].imshow(diff_image, cmap='hot')
            axes[2].set_title('差异图')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2], fraction=0.046)
        else:
            axes[2].text(0.5, 0.5, '形状不同\n无法计算差异',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('差异图')
            axes[2].axis('off')

    # 添加质量指标文本
    metrics_text = f"""重采样质量评估:

强度保持度:
  相关系数: {quality_metrics['intensity_preservation']['correlation']:.4f}
  均值变化: {quality_metrics['intensity_preservation']['mean_change_percent']:.2f}%

空间分辨率:
  体素大小变化: {quality_metrics['spatial_resolution']['voxel_size_change']:.3f}
  各向同性程度: {quality_metrics['spatial_resolution']['isotropic_degree']:.3f}

图像质量:
  信噪比: {quality_metrics['image_quality']['snr']:.2f}
  动态范围: {quality_metrics['image_quality']['dynamic_range']:.2f}
  对比度: {quality_metrics['image_quality']['contrast']:.3f}"""

    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

def compare_interpolation_methods(image: np.ndarray,
                                original_spacing: Tuple[float, ...],
                                target_spacing: Tuple[float, ...],
                                methods: List[str] = None,
                                save_path: Optional[str] = None):
    """
    比较不同插值方法的效果

    Args:
        image: 输入图像
        original_spacing: 原始间距
        target_spacing: 目标间距
        methods: 插值方法列表
        save_path: 保存路径
    """
    if methods is None:
        methods = ['nearest', 'linear', 'cubic']

    results = {}
    config = ResamplingConfig(target_spacing=target_spacing)
    resampler = MedicalImageResampler(config)

    fig, axes = plt.subplots(2, len(methods), figsize=(5*len(methods), 8))

    if len(methods) == 1:
        axes = axes.reshape(2, 1)

    # 选择中间切片
    if len(image.shape) == 3:
        z_mid = image.shape[0] // 2
        original_slice = image[z_mid, :, :]
    else:
        original_slice = image

    for i, method in enumerate(methods):
        try:
            # 重采样
            resampled_image, info = resampler.resample_image(
                image, original_spacing, target_spacing, method=method
            )

            # 获取重采样后的切片
            if len(resampled_image.shape) == 3:
                z_mid_resampled = resampled_image.shape[0] // 2
                resampled_slice = resampled_image[z_mid_resampled, :, :]
            else:
                resampled_slice = resampled_image

            results[method] = {
                'image': resampled_image,
                'info': info
            }

            # 显示结果
            im1 = axes[0, i].imshow(original_slice, cmap='gray')
            axes[0, i].set_title(f'原始图像\n{original_slice.shape}')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

            im2 = axes[1, i].imshow(resampled_slice, cmap='gray')
            axes[1, i].set_title(f'{method} 插值\n{resampled_slice.shape}')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

        except Exception as e:
            logger.error(f"插值方法 {method} 失败: {e}")
            axes[0, i].text(0.5, 0.5, f'{method}\n失败',
                           ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, f'{method}\n失败',
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[0, i].axis('off')
            axes[1, i].axis('off')

    plt.suptitle(f'插值方法比较\n原始间距: {original_spacing} -> 目标间距: {target_spacing}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"插值方法比较结果已保存: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()
    return results

def main():
    """主函数 - 演示医学图像重采样的完整流程"""
    print("="*80)
    print("医学图像重采样完整演示")
    print("="*80)

    # 1. 配置参数
    config = ResamplingConfig(
        target_spacing=(1.0, 1.0, 1.0),
        interpolation_method='linear',
        anti_aliasing=True,
        preserve_intensity=True
    )
    print(f"重采样配置: 目标间距={config.target_spacing}, 插值方法={config.interpolation_method}")

    # 2. 生成合成数据
    print("\n生成合成医学图像数据...")

    # 生成不同模态的图像
    modalities = ['ct', 'mri', 'pet']
    images = {}

    for modality in modalities:
        print(f"生成{modality.upper()}图像...")
        image, spacing = generate_synthetic_medical_image(
            shape=(64, 64, 32) if modality == 'pet' else (128, 128, 64),
            modality=modality
        )
        images[modality] = {
            'image': image,
            'spacing': spacing
        }
        print(f"  {modality.upper()}: 形状={image.shape}, 间距={spacing}")

    # 3. 执行重采样
    print("\n执行重采样...")
    resampler = MedicalImageResampler(config)
    results = []

    for modality, data in images.items():
        print(f"\n重采样{modality.upper()}图像...")

        original_image = data['image']
        original_spacing = data['spacing']

        # 定义目标间距
        target_spacing = (1.0, 1.0, 1.0)

        # 执行重采样
        resampled_image, resampling_info = resampler.resample_image(
            original_image, original_spacing, target_spacing
        )

        # 评估质量
        quality_metrics = evaluate_resampling_quality(
            original_image, resampled_image, original_spacing, target_spacing
        )

        result = {
            'modality': modality,
            'original_image': original_image,
            'resampled_image': resampled_image,
            'original_spacing': original_spacing,
            'resampled_spacing': target_spacing,
            'resampling_info': resampling_info,
            'quality_metrics': quality_metrics
        }
        results.append(result)

        print(f"  原始形状: {original_image.shape}")
        print(f"  重采样形状: {resampled_image.shape}")
        print(f"  强度相关系数: {quality_metrics['intensity_preservation']['correlation']:.4f}")
        print(f"  体素大小变化: {quality_metrics['spatial_resolution']['voxel_size_change']:.3f}")

    # 4. 比较插值方法
    print("\n比较不同插值方法...")

    # 使用CT图像比较插值方法
    ct_image = images['ct']['image']
    ct_spacing = images['ct']['spacing']
    target_spacing_large = (2.0, 2.0, 2.0)  # 下采样测试

    interpolation_results = compare_interpolation_methods(
        ct_image, ct_spacing, target_spacing_large,
        methods=['nearest', 'linear', 'cubic'],
        save_path="output/interpolation_comparison.png"
    )

    # 5. 可视化结果
    print("\n生成可视化结果...")
    os.makedirs("outputs", exist_ok=True)

    for i, result in enumerate(results):
        os.makedirs("output", exist_ok=True)
        save_path = f"output/resampling_result_{result['modality']}.png"
        visualize_resampling_results(
            result['original_image'],
            result['resampled_image'],
            result['original_spacing'],
            result['resampled_spacing'],
            result['quality_metrics'],
            save_path
        )

    # 6. 生成统计报告
    print("\n生成重采样报告...")

    report = {
        'configuration': {
            'target_spacing': config.target_spacing,
            'interpolation_method': config.interpolation_method,
            'anti_aliasing': config.anti_aliasing,
            'preserve_intensity': config.preserve_intensity
        },
        'results': []
    }

    for result in results:
        report['results'].append({
            'modality': result['modality'],
            'original_shape': result['original_image'].shape,
            'resampled_shape': result['resampled_image'].shape,
            'original_spacing': result['original_spacing'],
            'resampled_spacing': result['resampled_spacing'],
            'quality_metrics': result['quality_metrics'],
            'volume_change': float(np.prod(result['original_image'].shape) / np.prod(result['resampled_image'].shape))
        })

    # 保存报告
    os.makedirs("output", exist_ok=True)
    with open("output/resampling_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印总结
    print(f"\n{'='*80}")
    print("医学图像重采样演示完成!")
    print(f"{'='*80}")
    print(f"处理模态数量: {len(results)}")

    for result in results:
        modality = result['modality'].upper()
        correlation = result['quality_metrics']['intensity_preservation']['correlation']
        voxel_change = result['quality_metrics']['spatial_resolution']['voxel_size_change']
        print(f"{modality}: 强度相关系数={correlation:.4f}, 体素变化={voxel_change:.3f}")

    print(f"\n输出文件:")
    print(f"  - 重采样结果: output/resampling_result_*.png")
    print(f"  - 插值方法比较: output/interpolation_comparison.png")
    print(f"  - 重采样报告: output/resampling_report.json")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()