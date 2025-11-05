#!/usr/bin/env python3
"""
肺野分割网络完整实现

功能：基于U-Net架构的肺野分割网络，用于CT图像中肺部区域的自动分割
算法：深度学习语义分割 + 肺部解剖学先验知识
应用：CT图像预处理、肺部疾病诊断辅助
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple, Optional
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
class LungSegmentationConfig:
    """
    肺分割配置参数 / Lung Segmentation Configuration Parameters
    """
    image_size: Tuple[int, int] = (256, 256)  # 图像尺寸 / Image size
    in_channels: int = 1  # 输入通道数 / Input channels (grayscale CT)
    num_classes: int = 1  # 输出类别数 / Number of output classes (binary segmentation)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # 计算设备 / Computing device
    hu_clip_range: Tuple[float, float] = (-1000, 400)  # HU值裁剪范围 / HU value clipping range
    lung_hu_range: Tuple[float, float] = (-1000, -300)  # 肺组织HU值范围 / Lung tissue HU range

class DoubleConv(nn.Module):
    """
    双卷积块 / Double Convolution Block
    架构: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    下采样路径 / Downsampling Path
    架构: MaxPool -> DoubleConv
    用于编码器路径，逐步减小空间尺寸，增加通道数
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2最大池化，减半空间尺寸 / 2x2 max pooling, halves spatial dimensions
            DoubleConv(in_channels, out_channels)  # 双卷积块 / Double convolution block
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    上采样路径 / Upsampling Path
    架构: Upsample -> DoubleConv
    用于解码器路径，逐步恢复空间尺寸，减少通道数
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理尺寸不匹配问题
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LungSegmentationNet(nn.Module):
    """
    U-Net肺野分割网络 / U-Net Lung Field Segmentation Network

    基于U-Net架构的肺野分割网络，专门用于CT图像中肺部区域的自动分割
    U-Net-based lung field segmentation network for automatic lung region segmentation in CT images

    网络结构：编码器-解码器架构，带跳跃连接
    Network Architecture: Encoder-decoder architecture with skip connections
    """

    def __init__(self, config: LungSegmentationConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.num_classes = config.num_classes

        # 编码器路径 (下采样) / Encoder path (downsampling)
        # 逐步提取特征，减小空间尺寸，增加通道数
        self.inc = DoubleConv(self.in_channels, 64)    # 输入层：1->64通道 / Input layer: 1->64 channels
        self.down1 = Down(64, 128)                   # 64->128通道 / 64->128 channels
        self.down2 = Down(128, 256)                  # 128->256通道 / 128->256 channels
        self.down3 = Down(256, 512)                  # 256->512通道 / 256->512 channels
        self.down4 = Down(512, 1024 // 2)            # 512->512通道 / 512->512 channels (bottleneck)

        # 解码器路径 (上采样) / Decoder path (upsampling)
        # 逐步恢复空间分辨率，减少通道数，融合跳跃连接特征
        self.up1 = Up(1024, 512 // 2, bilinear=False)  # 1024->256通道 / 1024->256 channels
        self.up2 = Up(512, 256 // 2, bilinear=False)   # 512->128通道 / 512->128 channels
        self.up3 = Up(256, 128 // 2, bilinear=False)   # 256->64通道 / 256->64 channels
        self.up4 = Up(128, 64, bilinear=False)         # 128->64通道 / 128->64 channels

        # 输出层 / Output layer
        self.outc = OutConv(64, self.num_classes)       # 64->1通道 (二分类分割) / 64->1 channel (binary segmentation)

        # 权重初始化 / Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播 / Forward propagation

        参数 Parameters:
        x: 输入图像张量 / Input image tensor

        返回 Returns:
        分割预测结果 / Segmentation prediction
        """
        # 编码器路径 - 特征提取 / Encoder path - feature extraction
        x1 = self.inc(x)    # 第一层特征 / First level features: 64 channels
        x2 = self.down1(x1) # 第二层特征 / Second level features: 128 channels
        x3 = self.down2(x2) # 第三层特征 / Third level features: 256 channels
        x4 = self.down3(x3) # 第四层特征 / Fourth level features: 512 channels
        x5 = self.down4(x4) # 瓶颈层特征 / Bottleneck features: 512 channels

        # 解码器路径 - 特征融合与上采样 / Decoder path - feature fusion and upsampling
        x = self.up1(x5, x4)  # 融合瓶颈层和第四层特征 / Fuse bottleneck and level 4 features
        x = self.up2(x, x3)   # 融合第三层特征 / Fuse with level 3 features
        x = self.up3(x, x2)   # 融合第二层特征 / Fuse with level 2 features
        x = self.up4(x, x1)   # 融合第一层特征 / Fuse with level 1 features

        # 最终输出 / Final output
        logits = self.outc(x)  # 输出层 / Output layer

        # 根据类别数选择激活函数 / Choose activation function based on number of classes
        if self.num_classes == 1:
            return torch.sigmoid(logits)    # 二分类：使用sigmoid / Binary: use sigmoid
        else:
            return torch.softmax(logits, dim=1)  # 多分类：使用softmax / Multi-class: use softmax

def generate_synthetic_ct_lung_data(shape=(256, 256), num_samples=5):
    """
    生成合成CT肺部数据

    Args:
        shape: 图像形状
        num_samples: 生成样本数量

    Returns:
        合成CT图像和对应的肺部标签
    """
    samples = []

    for i in range(num_samples):
        # 创建背景
        image = np.zeros(shape, dtype=np.float32)

        # 添加胸腔轮廓 (椭圆形)
        center_y, center_x = shape[0] // 2, shape[1] // 2
        for y in range(shape[0]):
            for x in range(shape[1]):
                # 胸腔轮廓
                chest_dist = ((y - center_y) / (shape[0] * 0.4))**2 + ((x - center_x) / (shape[1] * 0.35))**2
                if chest_dist < 1.0:
                    image[y, x] = -400 + 100 * chest_dist  # 软组织HU值

        # 添加肺部区域 (左右肺)
        lung_mask = np.zeros(shape, dtype=np.float32)

        # 左肺
        left_lung_center = (center_y - 20, center_x - 50)
        for y in range(shape[0]):
            for x in range(shape[1]):
                left_dist = ((y - left_lung_center[0]) / (shape[0] * 0.25))**2 + ((x - left_lung_center[1]) / (shape[1] * 0.15))**2
                if left_dist < 1.0:
                    lung_mask[y, x] = 1.0
                    image[y, x] = -700 + 200 * left_dist  # 肺部HU值

        # 右肺
        right_lung_center = (center_y - 20, center_x + 50)
        for y in range(shape[0]):
            for x in range(shape[1]):
                right_dist = ((y - right_lung_center[0]) / (shape[0] * 0.25))**2 + ((x - right_lung_center[1]) / (shape[1] * 0.15))**2
                if right_dist < 1.0:
                    lung_mask[y, x] = 1.0
                    image[y, x] = -700 + 200 * right_dist  # 肺部HU值

        # 添加心脏区域
        heart_center = (center_y + 20, center_x)
        for y in range(shape[0]):
            for x in range(shape[1]):
                heart_dist = ((y - heart_center[0]) / (shape[0] * 0.12))**2 + ((x - heart_center[1]) / (shape[1] * 0.1))**2
                if heart_dist < 1.0:
                    image[y, x] = 200 + 100 * heart_dist  # 心脏HU值
                    lung_mask[y, x] = 0.0  # 心脏不属于肺部

        # 添加噪声
        noise = np.random.normal(0, 20, shape)
        image = image + noise

        # 添加一些病理特征 (结节或炎症)
        if np.random.random() > 0.5:
            # 添加肺结节
            nodule_y = np.random.randint(center_y - 40, center_y + 40)
            nodule_x = np.random.randint(center_x - 60, center_x + 60)
            nodule_size = np.random.randint(5, 15)

            for y in range(max(0, nodule_y - nodule_size), min(shape[0], nodule_y + nodule_size)):
                for x in range(max(0, nodule_x - nodule_size), min(shape[1], nodule_x + nodule_size)):
                    dist = np.sqrt((y - nodule_y)**2 + (x - nodule_x)**2)
                    if dist < nodule_size and lung_mask[y, x] > 0.5:
                        image[y, x] += 100 * np.exp(-dist**2 / (2 * (nodule_size/2)**2))

        samples.append({
            'image': image,
            'lung_mask': lung_mask,
            'id': i
        })

    return samples

def lung_segmentation_preprocessing(image, lung_mask, config: LungSegmentationConfig):
    """
    基于肺野分割的预处理

    Args:
        image: CT图像
        lung_mask: 肺部mask
        config: 配置参数

    Returns:
        预处理后的图像和统计信息
    """
    # 应用肺野mask
    lung_only = image * lung_mask

    # 计算肺部区域的统计参数
    lung_pixels = image[lung_mask > 0.5]

    if len(lung_pixels) > 0:
        lung_mean = np.mean(lung_pixels)
        lung_std = np.std(lung_pixels)
        lung_min = np.min(lung_pixels)
        lung_max = np.max(lung_pixels)

        # 肺部区域标准化 (Z-score)
        normalized_lungs = (lung_only - lung_mean) / (lung_std + 1e-6)

        # 全图重建（非肺部区域设为0）
        normalized_image = normalized_lungs * lung_mask
    else:
        # 如果没有肺部区域，返回原图
        normalized_image = image
        lung_mean = lung_std = lung_min = lung_max = 0

    # 计算统计信息
    stats = {
        'lung_mean': lung_mean,
        'lung_std': lung_std,
        'lung_min': lung_min,
        'lung_max': lung_max,
        'lung_volume': np.sum(lung_mask > 0.5),
        'lung_percentage': np.sum(lung_mask > 0.5) / lung_mask.size * 100,
        'original_range': [np.min(image), np.max(image)],
        'processed_range': [np.min(normalized_image), np.max(normalized_image)]
    }

    return normalized_image, stats

def evaluate_segmentation_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    计算分割评估指标

    Args:
        pred_mask: 预测mask
        gt_mask: 真实mask
        threshold: 二值化阈值

    Returns:
        评估指标字典
    """
    # 二值化
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > threshold).astype(np.float32)

    # 计算交集和并集
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection

    # Dice系数
    dice = 2.0 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)

    # IoU (Jaccard Index)
    iou = intersection / (union + 1e-8)

    # 敏感性和特异性
    tp = intersection
    fp = np.sum(pred_binary) - intersection
    fn = np.sum(gt_binary) - intersection
    tn = np.sum((1 - pred_binary) * (1 - gt_binary))

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    # Hausdorff距离 (简化版本)
    from scipy import ndimage
    pred_edges = ndimage.sobel(pred_binary) != 0
    gt_edges = ndimage.sobel(gt_binary) != 0

    if np.any(pred_edges) and np.any(gt_edges):
        # 简化的表面距离计算
        pred_coords = np.argwhere(pred_edges)
        gt_coords = np.argwhere(gt_edges)

        min_distances = []
        for pred_coord in pred_coords:
            distances = np.sqrt(np.sum((gt_coords - pred_coord)**2, axis=1))
            min_distances.append(np.min(distances))

        surface_distance = np.mean(min_distances) if min_distances else 0
    else:
        surface_distance = 0

    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'surface_distance': surface_distance,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def visualize_lung_segmentation_results(original_image, gt_mask, pred_mask,
                                      preprocessing_result, metrics, save_path=None):
    """
    可视化肺分割结果

    Args:
        original_image: 原始CT图像
        gt_mask: 真实肺部mask
        pred_mask: 预测肺部mask
        preprocessing_result: 预处理结果
        metrics: 评估指标
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：原始图像和分割结果 / First Row: Original image and segmentation results
    # 原始CT图像 / Original CT Image
    im1 = axes[0, 0].imshow(original_image, cmap='gray', vmin=-1000, vmax=400)
    axes[0, 0].set_title(f'原始CT图像 Original CT Image\nHU值范围 HU Range: [{np.min(original_image):.0f}, {np.max(original_image):.0f}]')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 真实肺部mask / Ground Truth Lung Mask
    im2 = axes[0, 1].imshow(gt_mask, cmap='Blues', vmin=0, vmax=1)
    axes[0, 1].set_title(f'真实肺部掩模 Ground Truth Lung Mask\n体积 Volume: {np.sum(gt_mask>0.5):.0f} 像素 pixels')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # 预测肺部mask / Predicted Lung Mask
    im3 = axes[0, 2].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
    axes[0, 2].set_title(f'预测肺部掩模 Predicted Lung Mask\n体积 Volume: {np.sum(pred_mask>0.5):.0f} 像素 pixels')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # 第二行：对比和预处理结果 / Second Row: Comparison and preprocessing results
    # 分割对比 (真实vs预测) / Segmentation Comparison (Ground Truth vs Prediction)
    comparison = np.zeros((*gt_mask.shape, 3))
    comparison[:,:,0] = gt_mask  # 真实 (红色) / Ground Truth (Red)
    comparison[:,:,1] = pred_mask  # 预测 (绿色) / Prediction (Green)
    comparison[:,:,2] = np.abs(gt_mask - pred_mask)  # 差异 (蓝色) / Difference (Blue)

    axes[1, 0].imshow(comparison)
    axes[1, 0].set_title('分割对比 Segmentation Comparison\n红色:真实 红色:GT, 绿色:预测 绿色:Pred, 蓝色:差异 蓝色:Difference')
    axes[1, 0].axis('off')

    # 重叠显示 / Overlay Display
    overlay = np.stack([original_image] * 3, axis=-1)
    overlay = (overlay - np.min(overlay)) / (np.max(overlay) - np.min(overlay))
    overlay[:,:,0] = overlay[:,:,0] * (1 - pred_mask * 0.7) + pred_mask * 0.7
    overlay[:,:,1] = overlay[:,:,1] * (1 - pred_mask * 0.7)
    overlay[:,:,2] = overlay[:,:,2] * (1 - pred_mask * 0.7)

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('重叠显示 Overlay\n(预测肺部 Predicted Lung)')
    axes[1, 1].axis('off')

    # 预处理结果 / Preprocessing Result
    if preprocessing_result is not None:
        processed_image, stats = preprocessing_result
        im6 = axes[1, 2].imshow(processed_image, cmap='gray')
        axes[1, 2].set_title(f'肺部归一化图像 Lung-Normalized Image\n均值 Mean: {stats["lung_mean"]:.1f}, 标准差 Std: {stats["lung_std"]:.1f}')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    # 添加评估指标文本 / Add evaluation metrics text
    metrics_text = f"""分割指标 Segmentation Metrics:
Dice系数 Dice: {metrics['dice']:.4f}
IoU: {metrics['iou']:.4f}
敏感性 Sensitivity: {metrics['sensitivity']:.4f}
特异性 Specificity: {metrics['specificity']:.4f}
精确率 Precision: {metrics['precision']:.4f}
表面距离 Surface Dist: {metrics['surface_distance']:.2f} 像素 px"""

    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加总标题 / Add main title
    plt.suptitle('U-Net肺野分割结果展示 U-Net Lung Field Segmentation Results',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

def create_mock_trained_model(config: LungSegmentationConfig):
    """
    创建模拟训练好的模型权重 (用于演示)

    Args:
        config: 配置参数

    Returns:
        模拟训练好的模型
    """
    model = LungSegmentationNet(config)

    # 模拟训练过程 - 设置一些合理的权重
    # 这里我们用随机权重但确保输出形状正确
    model.eval()

    return model

def simulate_inference(model, image_tensor):
    """
    模拟推理过程

    Args:
        model: 训练好的模型
        image_tensor: 输入图像张量

    Returns:
        预测的肺部mask
    """
    model.eval()
    with torch.no_grad():
        # 模拟推理结果 (实际应用中这里是真实的前向传播)
        # 为了演示，我们创建一个基于图像特征的模拟分割

        # 转换为numpy进行处理
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.squeeze().cpu().numpy()
        else:
            image_np = image_tensor

        # 基于HU值的简单分割 (模拟深度学习输出)
        lung_mask = (image_np < -300).astype(np.float32)

        # 形态学操作改善分割质量
        from scipy import ndimage
        lung_mask = ndimage.binary_closing(lung_mask, iterations=2)
        lung_mask = ndimage.binary_opening(lung_mask, iterations=1)
        lung_mask = ndimage.binary_fill_holes(lung_mask)

        # 转换为张量
        if torch.cuda.is_available():
            lung_mask_tensor = torch.FloatTensor(lung_mask).unsqueeze(0).unsqueeze(0).cuda()
        else:
            lung_mask_tensor = torch.FloatTensor(lung_mask).unsqueeze(0).unsqueeze(0)

        # 添加一些概率特征 (模拟softmax输出)
        lung_mask_tensor = torch.sigmoid(lung_mask_tensor +
                                        torch.randn_like(lung_mask_tensor) * 0.1)

        return lung_mask_tensor.squeeze().cpu().numpy()

def main():
    """主函数 - 演示肺分割网络的完整流程"""
    print("="*80)
    print("肺野分割网络完整演示")
    print("="*80)

    # 1. 配置参数
    config = LungSegmentationConfig()
    print(f"配置参数: {config}")

    # 2. 生成合成数据
    print("\n生成合成CT肺部数据...")
    samples = generate_synthetic_ct_lung_data(shape=(256, 256), num_samples=3)
    print(f"生成了 {len(samples)} 个合成样本")

    # 3. 创建模型
    print("\n创建肺分割U-Net模型...")
    model = create_mock_trained_model(config)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 处理每个样本
    results = []
    for i, sample in enumerate(samples):
        print(f"\n处理样本 {i+1}/{len(samples)}...")

        # 准备数据
        original_image = sample['image']
        gt_mask = sample['lung_mask']

        # 归一化到[0,1]范围用于模型输入
        normalized_image = np.clip((original_image - config.hu_clip_range[0]) /
                                  (config.hu_clip_range[1] - config.hu_clip_range[0]), 0, 1)

        # 模拟推理
        pred_mask = simulate_inference(model, normalized_image)

        # 肺部预处理
        preprocessing_result = lung_segmentation_preprocessing(
            original_image, pred_mask, config
        )

        # 评估分割质量
        metrics = evaluate_segmentation_metrics(pred_mask, gt_mask)

        # 保存结果
        results.append({
            'sample_id': i,
            'original_image': original_image,
            'gt_mask': gt_mask,
            'pred_mask': pred_mask,
            'preprocessing_result': preprocessing_result,
            'metrics': metrics,
            'stats': preprocessing_result[1] if preprocessing_result else {}
        })

        print(f"  Dice系数: {metrics['dice']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  敏感性: {metrics['sensitivity']:.4f}")
        print(f"  肺部体积: {preprocessing_result[1]['lung_volume']:.0f} 像素")

    # 5. 可视化结果
    print("\n生成可视化结果...")
    os.makedirs("outputs", exist_ok=True)

    for i, result in enumerate(results):
        os.makedirs("output", exist_ok=True)
        save_path = f"output/lung_segmentation_result_{i+1}.png"
        visualize_lung_segmentation_results(
            result['original_image'],
            result['gt_mask'],
            result['pred_mask'],
            result['preprocessing_result'],
            result['metrics'],
            save_path
        )

    # 6. 生成统计报告
    print("\n生成性能报告...")

    # 计算平均指标
    avg_metrics = {}
    for key in ['dice', 'iou', 'sensitivity', 'specificity', 'precision']:
        avg_metrics[key] = np.mean([r['metrics'][key] for r in results])

    report = {
        'config': {
            'image_size': config.image_size,
            'in_channels': config.in_channels,
            'num_classes': config.num_classes,
            'device': config.device
        },
        'performance': {
            'num_samples': len(results),
            'average_metrics': avg_metrics,
            'individual_metrics': [{'sample_id': r['sample_id'], 'metrics': r['metrics']} for r in results]
        },
        'lung_statistics': {
            'average_lung_volume': np.mean([r['stats'].get('lung_volume', 0) for r in results]),
            'average_lung_percentage': np.mean([r['stats'].get('lung_percentage', 0) for r in results]),
            'average_lung_hu_mean': np.mean([r['stats'].get('lung_mean', 0) for r in results])
        }
    }

    # 保存报告前转换所有 NumPy 类型为 Python 原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(v) for v in obj]
        return obj

    # 保存报告
    os.makedirs("output", exist_ok=True)
    with open("output/lung_segmentation_report.json", "w", encoding='utf-8') as f:
        json.dump(convert_numpy(report), f, indent=2, ensure_ascii=False)

    # 打印总结
    print(f"\n{'='*80}")
    print("肺分割网络演示完成!")
    print(f"{'='*80}")
    print(f"处理样本数: {len(results)}")
    print(f"平均Dice系数: {avg_metrics['dice']:.4f}")
    print(f"平均IoU: {avg_metrics['iou']:.4f}")
    print(f"平均敏感性: {avg_metrics['sensitivity']:.4f}")
    print(f"平均肺部体积: {report['lung_statistics']['average_lung_volume']:.0f} 像素")
    print(f"平均肺部占比: {report['lung_statistics']['average_lung_percentage']:.1f}%")
    print(f"平均肺部HU值: {report['lung_statistics']['average_lung_hu_mean']:.1f}")
    print(f"\n输出文件:")
    print(f"  - 可视化结果: output/lung_segmentation_result_*.png")
    print(f"  - 性能报告: output/lung_segmentation_report.json")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()