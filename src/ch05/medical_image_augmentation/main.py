#!/usr/bin/env python3
"""
通用医学图像增强完整实现 / Complete Medical Image Augmentation Implementation
功能：展示不同模态的医学图像增强技术，包括基础变换、高级技术和评估方法
Enhanced Features: Multi-modality augmentation, advanced techniques, evaluation metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import rotate, resize, AffineTransform, warp
from skimage.filters import gaussian
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# 设置中文字体和警告过滤 / Set Chinese font and warning filter
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# 过滤matplotlib警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

@dataclass
class AugmentationConfig:
    """增强配置参数 / Augmentation configuration parameters"""
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 4
    seed: int = 42

class MedicalImageAugmentation:
    """
    通用医学图像增强工具 / General Medical Image Augmentation Tool

    Features:
    - 多模态支持 (CT, MRI, X-ray)
    - 基础变换和高级技术
    - 医学约束验证
    - 质量评估指标
    """

    def __init__(self, config: AugmentationConfig):
        self.config = config
        np.random.seed(config.seed)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # 增强历史记录
        self.augmentation_history = []

    def create_sample_images(self):
        """
        创建不同模态的示例医学图像
        Create sample medical images for different modalities
        """
        print("创建不同模态的示例医学图像 / Creating sample medical images for different modalities...")

        images = {}
        masks = {}

        # 1. CT图像 - 肺部CT模拟
        ct_image = self._create_ct_sample()
        images['CT'] = ct_image
        masks['CT'] = None

        # 2. MRI图像 - 脑部MRI T1加权模拟
        mri_image = self._create_mri_sample()
        images['MRI'] = mri_image
        masks['MRI'] = None

        # 3. X光图像 - 胸部X光模拟
        xray_image = self._create_xray_sample()
        images['X-ray'] = xray_image
        masks['X-ray'] = None

        return images, masks

    def _create_ct_sample(self):
        """创建CT示例图像 / Create CT sample image"""
        # 创建512x512的CT图像
        image = np.zeros((512, 512), dtype=np.float32)

        # 模拟不同组织密度
        # 背景（空气）
        image[0:100, :] = -1000  # 空气密度

        # 软组织
        image[100:400, :] = np.linspace(-500, 300, 300)[:, np.newaxis]

        # 骨骼（肋骨）
        for i in range(5):
            y_pos = 150 + i * 30
            image[y_pos:y_pos+5, 100:400] = 1000  # 骨骼密度

        # 肺部（低密度区域）
        lung_y, lung_x = 200, 256
        lung_mask = ((np.arange(512)[:, np.newaxis] - lung_y)**2 +
                     (np.arange(512)[np.newaxis, :] - lung_x)**2) < 120**2
        image[lung_mask] = -800  # 肺部密度

        # 添加病灶
        nodule_y, nodule_x = 180, 200
        nodule_mask = ((np.arange(512)[:, np.newaxis] - nodule_y)**2 +
                       (np.arange(512)[np.newaxis, :] - nodule_x)**2) < 15**2
        image[nodule_mask] = -300  # 结节密度

        return image

    def _create_mri_sample(self):
        """创建MRI示例图像 / Create MRI sample image"""
        # 创建256x256的脑部MRI T1加权图像
        image = np.zeros((256, 256), dtype=np.float32)

        # 模拟脑组织结构
        y, x = np.ogrid[:256, :256]

        # 脑组织
        center_y, center_x = 128, 128
        brain_mask = ((x - center_x)**2 + (y - center_y)**2) < 100**2
        image[brain_mask] = 80 + np.random.randn(np.sum(brain_mask)) * 20

        # 脑室（低信号）
        ventricle_mask = ((x - center_x)**2 + (y - center_y)**2) < 30**2
        image[ventricle_mask] = 10 + np.random.randn(np.sum(ventricle_mask)) * 5

        # 白质（稍高信号）
        white_matter_mask = ((x - center_x)**2 + (y - center_y)**2) < 70**2
        white_matter_mask &= ~ventricle_mask
        image[white_matter_mask] = 100 + np.random.randn(np.sum(white_matter_mask)) * 15

        # 灰质（稍低信号）
        gray_matter_mask = brain_mask & ~white_matter_mask & ~ventricle_mask
        image[gray_matter_mask] = 70 + np.random.randn(np.sum(gray_matter_mask)) * 12

        return image

    def _create_xray_sample(self):
        """创建X光示例图像 / Create X-ray sample image"""
        # 创建512x256的胸部X光图像
        image = np.zeros((512, 256), dtype=np.float32)

        # 背景（空气）
        image[:, :] = 50

        # 模拟心脏（左下）
        heart_y, heart_x = 300, 100
        heart_size = (80, 60)
        heart_mask = (((np.arange(512)[:, np.newaxis] - heart_y)**2 / (heart_size[0]**2)) +
                     ((np.arange(256)[np.newaxis, :] - heart_x)**2 / (heart_size[1]**2))) < 1
        image[heart_mask] = 150  # 心脏密度

        # 模拟肺部（低密度）
        lung1_y, lung1_x = 150, 120
        lung1_mask = ((np.arange(512)[:, np.newaxis] - lung1_y)**2 / 90**2 +
                     (np.arange(256)[np.newaxis, :] - lung1_x)**2 / 70**2) < 1
        lung2_y, lung2_x = 150, 180
        lung2_mask = ((np.arange(512)[:, np.newaxis] - lung2_y)**2 / 90**2 +
                     (np.arange(256)[np.newaxis, :] - lung2_x)**2 / 70**2) < 1

        lung_mask = lung1_mask | lung2_mask
        image[lung_mask] = 100  # 肺部密度

        # 模拟肋骨
        for i in range(6):
            rib_y = 100 + i * 35
            image[rib_y:rib_y+3, 50:200] = 200  # 肋骨密度

        # 模拟横膈
        image[350, :] = 180

        return image

    def basic_augmentation(self, image: np.ndarray, modality: str) -> Dict[str, np.ndarray]:
        """
        基础医学图像增强 / Basic medical image augmentation
        """
        print(f"对{modality}图像应用基础增强 / Applying basic augmentation to {modality} image...")

        results = {}

        # 1. 旋转 - 医学约束：小角度旋转
        rotation_angles = self._get_rotation_angles(modality)
        for angle in rotation_angles:
            rotated = rotate(image, angle, preserve_range=True)
            results[f'rotation_{angle}'] = rotated

        # 2. 平移 - 小幅度平移
        translation_ranges = self._get_translation_ranges(modality)
        for dx, dy in translation_ranges:
            translated = self._translate_image(image, dx, dy)
            results[f'translation_{dx}_{dy}'] = translated

        # 3. 缩放 - 保持组织比例
        scale_factors = self._get_scale_factors(modality)
        for scale in scale_factors:
            scaled = self._scale_image(image, scale)
            results[f'scale_{scale}'] = scaled

        # 4. 翻转 - 考虑对称性
        flip_configs = self._get_flip_configs(modality)
        for flip_config in flip_configs:
            flipped = self._flip_image(image, flip_config)
            results[f'flip_{flip_config}'] = flipped

        return results

    def _get_rotation_angles(self, modality: str) -> List[float]:
        """获取适合的旋转角度 / Get suitable rotation angles"""
        if modality == 'CT':
            return [-5, 5, 10]  # CT：小角度旋转
        elif modality == 'MRI':
            return [-3, 3, 8]   # MRI：更小角度
        elif modality == 'X-ray':
            return [-2, 2, 5]   # X光：极小角度
        else:
            return [-5, 5, 10]

    def _get_translation_ranges(self, modality: str) -> List[Tuple[float, float]]:
        """获取适合的平移范围 / Get suitable translation ranges"""
        max_translation = {
            'CT': 0.05,
            'MRI': 0.03,
            'X-ray': 0.02
        }

        max_trans = max_translation.get(modality, 0.05)
        return [(max_trans, 0), (-max_trans, 0), (0, max_trans), (0, -max_trans)]

    def _get_scale_factors(self, modality: str) -> List[float]:
        """获取适合的缩放因子 / Get suitable scale factors"""
        if modality == 'CT':
            return [0.9, 1.1]
        elif modality == 'MRI':
            return [0.95, 1.05]
        elif modality == 'X-ray':
            return [0.98, 1.02]
        else:
            return [0.9, 1.1]

    def _get_flip_configs(self, modality: str) -> List[str]:
        """获取适合的翻转载置 / Get suitable flip configurations"""
        if modality in ['CT', 'X-ray']:
            return ['horizontal']  # CT和X光通常可以水平翻转
        elif modality == 'MRI':
            return ['horizontal', 'vertical']  # MRI可以考虑垂直翻转
        else:
            return ['horizontal']

    def _translate_image(self, image: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """图像平移 / Image translation"""
        # 使用更简单的方法实现平移
        dx_pixels = int(dx * image.shape[1])
        dy_pixels = int(dy * image.shape[0])
        
        # 创建平移后的图像
        translated = np.zeros_like(image)
        
        # 计算平移后的位置
        x_start = max(0, dx_pixels)
        x_end = min(image.shape[1], image.shape[1] + dx_pixels)
        y_start = max(0, dy_pixels)
        y_end = min(image.shape[0], image.shape[0] + dy_pixels)
        
        # 计算原始图像中的对应位置
        x_orig_start = max(0, -dx_pixels)
        x_orig_end = min(image.shape[1], image.shape[1] - dx_pixels)
        y_orig_start = max(0, -dy_pixels)
        y_orig_end = min(image.shape[0], image.shape[0] - dy_pixels)
        
        # 复制图像数据
        translated[y_start:y_end, x_start:x_end] = image[y_orig_start:y_orig_end, x_orig_start:x_orig_end]
        
        return translated

    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """图像缩放 / Image scaling"""
        scaled = resize(image,
                       (int(image.shape[0] * scale), int(image.shape[1] * scale)),
                       preserve_range=True,
                       anti_aliasing=True)
        # 确保返回的图像与原始图像尺寸相同
        if scaled.shape != image.shape:
            scaled = resize(scaled, image.shape, preserve_range=True, anti_aliasing=True)
        return scaled

    def _flip_image(self, image: np.ndarray, flip_config: str) -> np.ndarray:
        """图像翻转 / Image flipping"""
        if flip_config == 'horizontal':
            return np.fliplr(image)
        elif flip_config == 'vertical':
            return np.flipud(image)
        return image

    def intensity_augmentation(self, image: np.ndarray, modality: str) -> Dict[str, np.ndarray]:
        """
        强度增强 / Intensity augmentation
        """
        print(f"对{modality}图像应用强度增强 / Applying intensity augmentation to {modality} image...")

        results = {}

        # 1. 对比度调整
        contrast_factors = self._get_contrast_factors(modality)
        for factor in contrast_factors:
            adjusted = self._adjust_contrast(image, factor)
            results[f'contrast_{factor}'] = adjusted

        # 2. 亮度调整
        brightness_shifts = self._get_brightness_shifts(modality)
        for shift in brightness_shifts:
            adjusted = self._adjust_brightness(image, shift)
            results[f'brightness_{shift}'] = adjusted

        # 3. 噪声添加
        noise_configs = self._get_noise_configs(modality)
        for noise_config in noise_configs:
            noisy = self._add_noise(image, noise_config['type'], noise_config['level'])
            results[f'noise_{noise_config["type"]}_{noise_config["level"]}'] = noisy

        return results

    def _get_contrast_factors(self, modality: str) -> List[float]:
        """获取适合的对比度因子 / Get suitable contrast factors"""
        if modality == 'CT':
            return [0.8, 1.2, 1.5]
        elif modality == 'MRI':
            return [0.7, 1.0, 1.3]
        elif modality == 'X-ray':
            return [0.9, 1.1, 1.3]
        else:
            return [0.8, 1.2, 1.5]

    def _get_brightness_shifts(self, modality: str) -> List[float]:
        """获取适合的亮度偏移 / Get suitable brightness shifts"""
        if modality == 'CT':
            return [-50, 50, 100]
        elif modality == 'MRI':
            return [-30, 30, 60]
        elif modality == 'X-ray':
            return [-20, 20, 40]
        else:
            return [-50, 50, 100]

    def _get_noise_configs(self, modality: str) -> List[Dict]:
        """获取适合的噪声配置 / Get suitable noise configurations"""
        configs = []

        # 高斯噪声（电子噪声）
        configs.append({'type': 'gaussian', 'level': 10})
        configs.append({'type': 'gaussian', 'level': 20})

        # 泊松噪声（量子噪声）- 仅适用于CT
        if modality == 'CT':
            configs.append({'type': 'poisson', 'level': 5})
            configs.append({'type': 'poisson', 'level': 10})

        # 斑点噪声- 适用于超声，但这里也添加到其他模态用于研究
        configs.append({'type': 'speckle', 'level': 0.05})

        return configs

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """对比度调整 / Contrast adjustment"""
        mean = np.mean(image)
        return (image - mean) * factor + mean

    def _adjust_brightness(self, image: np.ndarray, shift: float) -> np.ndarray:
        """亮度调整 / Brightness adjustment"""
        return image + shift

    def _add_noise(self, image: np.ndarray, noise_type: str, level: float) -> np.ndarray:
        """添加噪声 / Add noise"""
        if noise_type == 'gaussian':
            return image + np.random.normal(0, level, image.shape)
        elif noise_type == 'poisson':
            # 泊松噪声（需要正数）
            image_positive = image - np.min(image)
            noisy = np.random.poisson(image_positive * level / 10) / (level / 10)
            result = (noisy / np.max(image_positive) * np.max(image_positive) +
                     np.min(image_positive))
            # 确保返回的是二维数组
            return result.reshape(image.shape)
        elif noise_type == 'speckle':
            noise = np.random.randn(*image.shape)
            return image + image * noise * level
        else:
            return image

    def advanced_augmentation(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        高级增强技术 / Advanced augmentation techniques
        """
        print("应用高级增强技术 / Applying advanced augmentation techniques...")

        results = {}

        # 1. 弹性变形
        elastic_images = self._elastic_deformation(image, mask, mask is not None)
        for i, (aug_img, aug_mask) in enumerate(elastic_images):
            if mask is not None:
                results[f'elastic_deformation_{i+1}'] = aug_img
                results[f'elastic_deformation_{i+1}_mask'] = aug_mask
            else:
                results[f'elastic_deformation_{i+1}'] = aug_img

        # 2. 混合增强（CutMix）
        cutmix_images = self._cutmix(image, image if mask is None else image, mask, mask is not None)
        for i, (aug_img, aug_mask) in enumerate(cutmix_images):
            if mask is not None:
                results[f'cutmix_{i+1}'] = aug_img
                results[f'cutmix_{i+1}_mask'] = aug_mask
            else:
                results[f'cutmix_{i+1}'] = aug_img

        # 3. 局部遮挡
        occlusion_images = self._partial_occlusion(image, mask, mask is not None)
        for i, (aug_img, aug_mask) in enumerate(occlusion_images):
            if mask is not None:
                results[f'occlusion_{i+1}'] = aug_img
                results[f'occlusion_{i+1}_mask'] = aug_mask
            else:
                results[f'occlusion_{i+1}'] = aug_img

        return results

    def _elastic_deformation(self, image: np.ndarray, mask: Optional[np.ndarray],
                             has_mask: bool) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """弹性变形 / Elastic deformation"""
        results = []

        for _ in range(2):  # 生成两个不同的变形
            # 生成随机位移场
            shape = image.shape
            alpha = np.random.uniform(800, 1200)
            sigma = np.random.uniform(6, 10)

            dx = gaussian(np.random.randn(*shape), sigma, mode='reflect') * alpha
            dy = gaussian(np.random.randn(*shape), sigma, mode='reflect') * alpha

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0], dtype=np.float32))
            indices = np.array([y + dy, x + dx])

            # 应用变形
            warped_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
            warped_image = warped_image.reshape(shape)

            if has_mask:
                warped_mask = ndimage.map_coordinates(mask, indices, order=0, mode='reflect')
                warped_mask = warped_mask.reshape(shape)
                results.append((warped_image, warped_mask))
            else:
                results.append((warped_image, None))

        return results

    def _cutmix(self, image1: np.ndarray, image2: np.ndarray, mask: Optional[np.ndarray],
                has_mask: bool) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """CutMix增强 / CutMix augmentation"""
        results = []

        for _ in range(2):  # 生成两个不同的CutMix
            lam = np.random.beta(1.0, 1.0)

            # 生成随机矩形区域
            cut_ratio = np.sqrt(1.0 - lam)
            cut_w = int(image1.shape[1] * cut_ratio)
            cut_h = int(image1.shape[0] * cut_ratio)

            # 随机位置
            cx = np.random.randint(cut_w // 2, image1.shape[1] - cut_w // 2)
            cy = np.random.randint(cut_h // 2, image1.shape[0] - cut_h // 2)

            bbx1 = np.clip(cx - cut_w // 2, 0, image1.shape[1])
            bby1 = np.clip(cy - cut_h // 2, 0, image1.shape[0])
            bbx2 = np.clip(cx + cut_w // 2, 0, image1.shape[1])
            bby2 = np.clip(cy + cut_h // 2, 0, image1.shape[0])

            # 创建混合图像
            mixed_image = image1.copy()
            mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

            if has_mask:
                mixed_mask = mask.copy()
                mixed_mask[bby1:bby2, bbx1:bbx2] = mask[bby1:bby2, bbx1:bbx2]
                results.append((mixed_image, mixed_mask))
            else:
                results.append((mixed_image, None))

        return results

    def _partial_occlusion(self, image: np.ndarray, mask: Optional[np.ndarray],
                         has_mask: bool) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """局部遮挡 / Partial occlusion"""
        results = []

        for _ in range(2):  # 生成两个不同的遮挡
            # 随机矩形遮挡
            occ_h = np.random.randint(10, image.shape[0] // 4)
            occ_w = np.random.randint(10, image.shape[1] // 4)
            occ_y = np.random.randint(0, image.shape[0] - occ_h)
            occ_x = np.random.randint(0, image.shape[1] - occ_w)

            # 创建遮挡图像
            occluded_image = image.copy()
            occluded_image[occ_y:occ_y+occ_h, occ_x:occ_x+occ_w] = np.min(image)

            if has_mask:
                occluded_mask = mask.copy()
                results.append((occluded_image, occluded_mask))
            else:
                results.append((occluded_image, None))

        return results

    def evaluate_augmentation(self, original: np.ndarray, augmented: np.ndarray,
                            method: str = 'psnr') -> Dict[str, float]:
        """
        评估增强效果 / Evaluate augmentation effect
        """
        print(f"使用{method}方法评估增强效果 / Evaluating augmentation effect using {method} method...")

        metrics = {}

        if method in ['psnr', 'mse', 'all']:
            # MSE
            mse = np.mean((original - augmented) ** 2)
            metrics['mse'] = float(mse)

            # PSNR
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                metrics['psnr'] = float(psnr)
            else:
                metrics['psnr'] = float('inf')

        if method in ['ssim', 'all']:
            try:
                from skimage.metrics import structural_similarity as ssim

                # 归一化图像到0-255范围进行SSIM计算
                original_norm = self._normalize_for_ssim(original)
                augmented_norm = self._normalize_for_ssim(augmented)

                # 检查归一化结果
                if original_norm.size == 0 or augmented_norm.size == 0:
                    metrics['ssim'] = 'N/A'
                    if method == 'all':
                        print("  Debug: Empty image after normalization")
                else:
                    # 检查图像是否有效
                    orig_unique = len(np.unique(original_norm))
                    aug_unique = len(np.unique(augmented_norm))

                    if orig_unique == 1 and aug_unique == 1:
                        # 两张图像都是均匀的，SSIM应该为1（完全相同）或0（不同）
                        if np.array_equal(original_norm, augmented_norm):
                            metrics['ssim'] = 1.0
                        else:
                            metrics['ssim'] = 0.0
                        if method == 'all':
                            print(f"  Debug: Uniform images - orig_unique={orig_unique}, aug_unique={aug_unique}, ssim={metrics['ssim']}")
                    else:
                        # 正常情况计算SSIM
                        try:
                            if original_norm.shape == augmented_norm.shape:
                                ssim_value = ssim(original_norm, augmented_norm, data_range=255)
                                if not np.isnan(ssim_value) and not np.isinf(ssim_value):
                                    metrics['ssim'] = float(ssim_value)
                                else:
                                    metrics['ssim'] = 'N/A'
                                    if method == 'all':
                                        print(f"  Debug: SSIM calculation returned NaN/Inf: {ssim_value}")
                            else:
                                # 如果形状不匹配，调整到相同尺寸
                                from skimage.transform import resize
                                if original_norm.size > augmented_norm.size:
                                    resized_aug = resize(augmented_norm, original_norm.shape, preserve_range=True, anti_aliasing=False)
                                    ssim_value = ssim(original_norm, resized_aug.astype(np.uint8), data_range=255)
                                else:
                                    resized_orig = resize(original_norm, augmented_norm.shape, preserve_range=True, anti_aliasing=False)
                                    ssim_value = ssim(resized_orig.astype(np.uint8), augmented_norm, data_range=255)

                                if not np.isnan(ssim_value) and not np.isinf(ssim_value):
                                    metrics['ssim'] = float(ssim_value)
                                else:
                                    metrics['ssim'] = 'N/A'
                                    if method == 'all':
                                        print(f"  Debug: SSIM calculation after resize returned NaN/Inf: {ssim_value}")
                        except Exception as ssim_err:
                            metrics['ssim'] = 'N/A'
                            if method == 'all':
                                print(f"  Debug: SSIM calculation error: {ssim_err}")

            except ImportError:
                warnings.warn("scikit-image not available for SSIM calculation")
                metrics['ssim'] = 'N/A'
            except Exception as e:
                warnings.warn(f"SSIM calculation failed: {e}")
                metrics['ssim'] = 'N/A'

        if method in ['histogram', 'all']:
            # 直方图相关性 - 使用实际图像的数值范围
            min_val = min(np.min(original), np.min(augmented))
            max_val = max(np.max(original), np.max(augmented))

            hist_orig, _ = np.histogram(original, bins=256, range=(min_val, max_val))
            hist_aug, _ = np.histogram(augmented, bins=256, range=(min_val, max_val))

            # 避免除零
            hist_orig = hist_orig.astype(float) + 1e-10
            hist_aug = hist_aug.astype(float) + 1e-10

            # 相关性
            correlation = np.corrcoef(hist_orig, hist_aug)[0, 1]
            if not np.isnan(correlation):
                metrics['histogram_correlation'] = float(correlation)
            else:
                metrics['histogram_correlation'] = 0.0

        return metrics

    def _normalize_for_ssim(self, image: np.ndarray) -> np.ndarray:
        """归一化图像到0-255范围用于SSIM计算 / Normalize image to 0-255 range for SSIM calculation"""
        # 检查图像是否为空或无效
        if image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        # 线性归一化到0-255范围
        img_min = np.min(image)
        img_max = np.max(image)

        if img_max - img_min > 1e-10:
            # 正常情况：线性归一化
            normalized = (image - img_min) / (img_max - img_min) * 255.0
        else:
            # 边缘情况：所有像素值相同
            # 为避免全零图像导致SSIM计算问题，创建一个小的对比度
            normalized = np.full_like(image, 128.0)  # 使用中间值

        return np.clip(normalized, 0, 255).astype(np.uint8)

    def visualize_augmentation_results(self, original_images: Dict[str, np.ndarray],
                                      basic_results: Dict[str, Dict[str, np.ndarray]],
                                      advanced_results: Dict[str, Dict[str, np.ndarray]]):
        """
        纯图像可视化，无文字叠加 - 统一格式
        Pure image visualization without text overlays - consistent format
        """
        print("Generating augmentation visualization...")

        modality = list(original_images.keys())[0] if original_images else 'Unknown'

        # 创建图像展示所有结果 - 使用与simple版本相同的布局
        fig = plt.figure(figsize=(16, 10))

        fig.suptitle(f'{modality} Medical Image Augmentation Demo', fontsize=16, fontweight='bold')

        # 原始图像
        ax1 = plt.subplot(3, 5, 1)
        ax1.imshow(original_images[modality], cmap='gray')
        ax1.set_title('Original', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 显示基础增强结果
        basic_types = ['rotation', 'translation', 'scale', 'flip']
        basic_names = ['Rotation', 'Translation', 'Scale', 'Flip']

        for i, (basic_type, basic_name) in enumerate(zip(basic_types, basic_names)):
            ax = plt.subplot(3, 5, i + 2)

            # 选择该类型的第一个增强结果
            keys = [k for k in basic_results.keys() if basic_type in k]
            if keys:
                aug_image = basic_results[keys[0]]
                # 检查是否是元组（图像和掩码），如果是则只取图像
                if isinstance(aug_image, tuple):
                    aug_image = aug_image[0]
                ax.imshow(aug_image, cmap='gray')
                ax.set_title(f'{basic_name}', fontsize=10)
            else:
                ax.set_title(f'{basic_name} (N/A)', fontsize=10)
            ax.axis('off')

        # 显示强度增强结果 (从basic_results中提取)
        intensity_types = ['contrast', 'brightness', 'noise_gaussian', 'noise_poisson']
        intensity_names = ['Contrast', 'Brightness', 'Gaussian Noise', 'Poisson Noise']

        for i, (intensity_type, intensity_name) in enumerate(zip(intensity_types, intensity_names)):
            ax = plt.subplot(3, 5, i + 6)

            # 选择该类型的第一个增强结果
            keys = [k for k in basic_results.keys() if intensity_type in k]
            if keys:
                aug_image = basic_results[keys[0]]
                # 检查是否是元组（图像和掩码），如果是则只取图像
                if isinstance(aug_image, tuple):
                    aug_image = aug_image[0]
                ax.imshow(aug_image, cmap='gray')
                ax.set_title(f'{intensity_name}', fontsize=10)
            else:
                ax.set_title(f'{intensity_name} (N/A)', fontsize=10)
            ax.axis('off')

        # 显示高级增强结果
        advanced_types = ['elastic_deformation', 'cutmix', 'occlusion']
        advanced_names = ['Elastic', 'CutMix', 'Occlusion']

        for i, (advanced_type, advanced_name) in enumerate(zip(advanced_types, advanced_names)):
            ax = plt.subplot(3, 5, i + 10)

            # 选择该类型的第一个增强结果
            keys = [k for k in advanced_results.keys() if advanced_type in k]
            if keys:
                aug_image = advanced_results[keys[0]]
                # 检查是否是元组（图像和掩码），如果是则只取图像
                if isinstance(aug_image, tuple):
                    aug_image = aug_image[0]
                ax.imshow(aug_image, cmap='gray')
                ax.set_title(f'{advanced_name}', fontsize=10)
            else:
                ax.set_title(f'{advanced_name} (N/A)', fontsize=10)
            ax.axis('off')

        # 第15个位置留空
        ax_empty = plt.subplot(3, 5, 15)
        ax_empty.axis('off')

        # 在终端打印评估指标和统计信息
        print(f"\n{'='*60}")
        print(f"Medical Image Augmentation - Quality Evaluation:")
        print(f"{'='*60}")
        print(f"Modality Type: {modality}")
        print(f"Image Size: {original_images[modality].shape}")
        print(f"Pixel Range: [{original_images[modality].min():.1f}, {original_images[modality].max():.1f}]")

        # 计算技术数量
        basic_count = len([k for k in basic_results.keys() if any(t in k for t in basic_types)])
        intensity_count = len([k for k in basic_results.keys() if any(t in k for t in intensity_types)])
        advanced_count = len(advanced_results)
        total_count = len(basic_results) + len(advanced_results)

        print(f"Basic Augmentation: {basic_count} techniques")
        print(f"Intensity Augmentation: {intensity_count} techniques")
        print(f"Advanced Augmentation: {advanced_count} techniques")
        print(f"Total Techniques: {total_count}")

        # 质量评估指标汇总
        psnr_values = []
        ssim_values = []

        for basic_type in basic_types:
            keys = [k for k in basic_results.keys() if basic_type in k]
            if keys:
                aug_image = basic_results[keys[0]]
                if isinstance(aug_image, tuple):
                    aug_image = aug_image[0]
                try:
                    metrics = self.evaluate_augmentation(original_images[modality], aug_image, method='all')
                    psnr_val = metrics.get('psnr', 0)
                    ssim_val = metrics.get('ssim', 0)
                    if isinstance(psnr_val, (int, float)) and not np.isinf(psnr_val):
                        psnr_values.append(psnr_val)
                    if isinstance(ssim_val, (int, float)):
                        ssim_values.append(ssim_val)
                except:
                    pass

        if psnr_values:
            avg_psnr = np.mean(psnr_values)
            print(f"\nQuality Metrics Summary:")
            print(f"  Average PSNR: {avg_psnr:.2f} dB")
        if ssim_values:
            avg_ssim = np.mean(ssim_values)
            print(f"  Average SSIM: {avg_ssim:.3f}")

        print(f"{'='*60}")

        plt.tight_layout()

        # 保存图像 - 使用统一的文件名格式，无字体问题
        output_path = self.output_dir / f"medical_image_augmentation_{modality.lower()}_demo.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\nVisualization saved: {output_path}")

        plt.close()

        # 在终端打印评估指标
        print("\n" + "="*80)
        print("增强效果评估指标 / Augmentation Evaluation Metrics:")
        print("="*80)

        for basic_type in basic_types:
            keys = [k for k in basic_results.keys() if basic_type in k]
            if keys:
                aug_image = basic_results[keys[0]]
                # 检查是否是元组（图像和掩码），如果是则只取图像
                if isinstance(aug_image, tuple):
                    aug_image = aug_image[0]

                # 确保图像有效
                if aug_image is not None and aug_image.size > 0:
                    try:
                        metrics = self.evaluate_augmentation(original_images[modality], aug_image, method='all')
                        basic_name = basic_names[basic_types.index(basic_type)]
                        print(f"\n{basic_name}:")

                        psnr_val = metrics.get('psnr', 'N/A')
                        if isinstance(psnr_val, (int, float)) and not np.isinf(psnr_val):
                            print(f"  PSNR: {psnr_val:.2f} dB")
                        else:
                            print(f"  PSNR: {psnr_val}")

                        ssim_val = metrics.get('ssim', 'N/A')
                        if isinstance(ssim_val, (int, float)):
                            print(f"  SSIM: {ssim_val:.3f}")
                        else:
                            print(f"  SSIM: {ssim_val}")
                    except Exception as e:
                        print(f"\n{basic_name}:")
                        print(f"  评估失败 / Evaluation failed: {e}")
                        print(f"  PSNR: N/A")
                        print(f"  SSIM: N/A")

        # 打印技术参数
        print("\n" + "="*80)
        print("技术参数说明 / Technical Parameters:")
        print("="*80)
        print(f"模态 Modality: {modality}")
        print(f"图像尺寸 Image Size: {original_images[modality].shape}")
        print(f"基础增强 Basic Augmentation: {len(basic_results)} 种")
        print(f"高级增强 Advanced Augmentation: {len(advanced_results)} 种")
        print(f"总增强技术 Total Augmentation: {len(basic_results) + len(advanced_results)} 种")

        return str(output_path)

    def main(self):
        """
        主函数 / Main function
        """
        print("=" * 80)
        print("通用医学图像增强完整演示 / Complete Medical Image Augmentation Demo")
        print("=" * 80)

        # 创建示例图像
        print("\n[Medical] 创建不同模态的示例医学图像 / Creating sample medical images for different modalities...")
        original_images, masks = self.create_sample_images()

        # 选择一个模态进行详细演示
        modality = 'CT'  # 可以选择 'MRI' 或 'X-ray'
        image = original_images[modality]
        mask = masks[modality]

        print(f"\n[Select] 选择{modality}图像进行详细演示 / Selected {modality} image for detailed demonstration...")
        print(f"图像尺寸 / Image size: {image.shape}")
        print(f"像素值范围 / Pixel range: [{image.min():.1f}, {image.max():.1f}]")

        # 应用基础增强
        print(f"\n[Basic] 应用基础增强技术 / Applying basic augmentation techniques...")
        basic_results = self.basic_augmentation(image, modality)
        print(f"生成 {len(basic_results)} 种基础增强效果 / Generated {len(basic_results)} basic augmentation effects")

        # 应用强度增强
        print(f"\n[Process] 应用强度增强技术 / Applying intensity augmentation techniques...")
        intensity_results = self.intensity_augmentation(image, modality)
        print(f"生成 {len(intensity_results)} 种强度增强效果 / Generated {len(intensity_results)} intensity augmentation effects")

        # 合并基础和强度增强结果
        all_basic_results = {**basic_results, **intensity_results}

        # 应用高级增强
        print(f"\n[Advanced] 应用高级增强技术 / Applying advanced augmentation techniques...")
        advanced_results = self.advanced_augmentation(image, mask)
        print(f"生成 {len(advanced_results)} 种高级增强效果 / Generated {len(advanced_results)} advanced augmentation effects")

        # 可视化结果
        print(f"\n[Visualize] 生成增强效果可视化 / Generating augmentation visualization...")
        viz_path = self.visualize_augmentation_results(
            original_images, all_basic_results, advanced_results
        )

        # 保存报告
        self._save_report(modality, len(basic_results), len(advanced_results))

        print(f"\n[Report] 生成完整报告 / Generating complete report...")

        return {
            'modality': modality,
            'original_image_shape': image.shape,
            'basic_augmentation_count': len(basic_results),
            'intensity_augmentation_count': len(intensity_results),
            'advanced_augmentation_count': len(advanced_results),
            'visualization_path': viz_path,
            'total_augmentation_count': len(basic_results) + len(intensity_results) + len(advanced_results)
        }

    def _save_report(self, modality: str, basic_count: int, advanced_count: int):
        """保存增强报告 / Save augmentation report"""
        report = {
            'timestamp': '2025-11-10',
            'modality': modality,
            'statistics': {
                'basic_augmentation_count': basic_count,
                'intensity_augmentation_count': basic_count,  # intensity is part of basic
                'advanced_augmentation_count': advanced_count,
                'total_augmentation_count': basic_count + advanced_count
            },
            'techniques_applied': {
                'basic': [
                    'rotation', 'translation', 'scale', 'flip',
                    'contrast_adjustment', 'brightness_adjustment',
                    'gaussian_noise', 'poisson_noise', 'speckle_noise'
                ],
                'advanced': [
                    'elastic_deformation', 'cutmix', 'partial_occlusion'
                ]
            }
        }

        report_path = self.output_dir / "augmentation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"报告已保存到 / Report saved to: {report_path}")

if __name__ == "__main__":
    # 创建配置
    config = AugmentationConfig()

    # 创建增强器并运行演示
    augmentor = MedicalImageAugmentation(config)
    results = augmentor.main()

    print("\n" + "=" * 80)
    print("通用医学图像增强演示完成 / Medical Image Augmentation Demo Completed!")
    print("=" * 80)
    print(f"模态 / Modality: {results['modality']}")
    print(f"图像尺寸 / Image shape: {results['original_image_shape']}")
    print(f"增强技术总数 / Total augmentation techniques: {results['total_augmentation_count']}")
    print(f"可视化文件 / Visualization file: {results['visualization_path']}")