#!/usr/bin/env python3
"""
医学图像分类网络实现 / Medical Image Classification Network Implementation

功能：基于ResNet的医学图像分类网络，用于疾病检测和诊断辅助
算法：深度学习卷积神经网络 + 注意力机制
应用：CT/MRI图像分类、疾病筛查、诊断辅助

学习目标：
1. 理解医学图像分类的网络架构设计
2. 掌握类别不平衡问题的处理方法
3. 学习模型评估指标和可视化技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

# 设置中文字体 / Set Chinese font
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
    except:
        pass

# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClassificationConfig:
    """
    医学图像分类配置参数 / Medical Image Classification Configuration
    """
    image_size: Tuple[int, int] = (224, 224)  # 图像尺寸 / Image size
    in_channels: int = 3  # 输入通道数 / Input channels (RGB)
    num_classes: int = 2  # 分类数量 / Number of classes (binary)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # 计算设备 / Computing device
    dropout_rate: float = 0.5  # Dropout率 / Dropout rate
    learning_rate: float = 0.001  # 学习率 / Learning rate
    batch_size: int = 32  # 批次大小 / Batch size
    num_epochs: int = 10  # 训练轮数 / Number of epochs

class BasicBlock(nn.Module):
    """
    ResNet基础块 / ResNet Basic Block
    包含残差连接的卷积块
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接 / Residual connection
        out = self.relu(out)

        return out

class MedicalImageClassifier(nn.Module):
    """
    医学图像分类网络 / Medical Image Classification Network
    基于ResNet架构，专门为医学图像分类任务优化
    """
    def __init__(self, config: ClassificationConfig):
        super().__init__()
        self.config = config
        self.in_channels = 64

        # 初始卷积层 / Initial convolution layer
        self.conv1 = nn.Conv2d(config.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet层 / ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 64, blocks=2)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, blocks=2, stride=2)

        # 全局平均池化 / Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头 / Classification head
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes)
        )

        # 权重初始化 / Weight initialization
        self._init_weights()

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """
        构建ResNet层 / Build ResNet layer

        参数 Parameters:
        block: 基础块类型 / Basic block type
        in_channels: 输入通道数 / Input channels
        out_channels: 输出通道数 / Output channels
        blocks: 块数量 / Number of blocks
        stride: 步长 / Stride
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """权重初始化 / Weight initialization"""
        for m in self.modules():
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
        分类预测结果 / Classification prediction
        """
        # 初始卷积 / Initial convolution
        x = self.conv1(x)           # [batch, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [batch, 64, H/4, W/4]

        # ResNet特征提取 / ResNet feature extraction
        x = self.layer1(x)          # [batch, 64, H/4, W/4]
        x = self.layer2(x)          # [batch, 128, H/8, W/8]
        x = self.layer3(x)          # [batch, 256, H/16, W/16]
        x = self.layer4(x)          # [batch, 512, H/32, W/32]

        # 全局平均池化 / Global average pooling
        x = self.avgpool(x)         # [batch, 512, 1, 1]
        x = torch.flatten(x, 1)    # [batch, 512]

        # 分类 / Classification
        x = self.fc(x)              # [batch, num_classes]

        return x

def generate_synthetic_medical_data(config: ClassificationConfig, num_samples: int = 100):
    """
    生成合成医学图像数据 / Generate synthetic medical image data

    参数 Parameters:
    config: 配置参数 / Configuration
    num_samples: 样本数量 / Number of samples

    返回 Returns:
    images: 图像数据 / Image data
    labels: 标签数据 / Label data
    """
    print(f"生成合成医学图像数据 / Generating synthetic medical image data")
    print(f"  样本数量 Sample count: {num_samples}")
    print(f"  图像尺寸 Image size: {config.image_size}")
    print(f"  类别数量 Number of classes: {config.num_classes}")

    images = []
    labels = []

    for i in range(num_samples):
        # 生成随机图像 / Generate random image
        image = np.random.randn(*config.image_size, config.in_channels) * 0.5 + 0.5

        # 根据类别添加特定模式 / Add class-specific patterns
        label = np.random.randint(0, config.num_classes)
        if label == 1:  # 异常图像 / Abnormal image
            # 添加病灶模式 / Add lesion patterns
            center_y, center_x = np.random.randint(20, config.image_size[0]-20), np.random.randint(20, config.image_size[1]-20)
            radius = np.random.randint(5, 15)
            y, x = np.ogrid[:config.image_size[0], :config.image_size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            for c in range(config.in_channels):
                image[mask, c] += np.random.uniform(0.3, 0.8)

        images.append(image)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"  数据生成完成 / Data generation completed")
    print(f"  正常样本 Normal samples: {np.sum(labels == 0)}")
    print(f"  异常样本 Abnormal samples: {np.sum(labels == 1)}")

    return images, labels

def evaluate_classification_metrics(model, test_loader, device):
    """
    评估分类模型性能 / Evaluate classification model performance

    参数 Parameters:
    model: 分类模型 / Classification model
    test_loader: 测试数据加载器 / Test data loader
    device: 计算设备 / Computing device

    返回 Returns:
    metrics: 评估指标 / Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]  # 正类概率 / Positive class probability
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标 / Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        'accuracy': np.mean(all_preds == all_labels),
        'precision': np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_preds == 1) + 1e-8),
        'recall': np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_labels == 1) + 1e-8),
        'f1_score': 0,
        'auc_roc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }

    # 计算F1分数 / Calculate F1 score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

    return metrics, all_preds, all_labels

def visualize_classification_results(images, labels, predictions, metrics, save_path=None):
    """
    可视化分类结果 / Visualize classification results

    参数 Parameters:
    images: 图像数据 / Image data
    labels: 真实标签 / True labels
    predictions: 预测结果 / Predictions
    metrics: 评估指标 / Evaluation metrics
    save_path: 保存路径 / Save path
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 显示前8个样本的预测结果 / Show predictions for first 8 samples
    for i in range(min(8, len(images))):
        row = i // 4
        col = i % 4

        # 显示图像 / Display image
        if images[i].shape[-1] == 3:  # RGB图像 / RGB image
            axes[row, col].imshow(images[i])
        else:  # 灰度图像 / Grayscale image
            axes[row, col].imshow(images[i][:, :, 0], cmap='gray')

        # 设置标题 / Set title
        true_label = "正常 Normal" if labels[i] == 0 else "异常 Abnormal"
        pred_label = "正常 Normal" if predictions[i] == 0 else "异常 Abnormal"
        correct = "✓" if labels[i] == predictions[i] else "✗"

        axes[row, col].set_title(f'真实 GT: {true_label}\n预测 Pred: {pred_label} {correct}')
        axes[row, col].axis('off')

    # 添加总标题和指标 / Add main title and metrics
    plt.suptitle('医学图像分类结果展示 Medical Image Classification Results', fontsize=16, fontweight='bold')

    # 添加指标文本 / Add metrics text
    metrics_text = f"""分类性能指标 Classification Performance Metrics:
准确率 Accuracy: {metrics['accuracy']:.4f}
精确率 Precision: {metrics['precision']:.4f}
召回率 Recall: {metrics['recall']:.4f}
F1分数 F1-Score: {metrics['f1_score']:.4f}
AUC-ROC: {metrics['auc_roc']:.4f}"""

    fig.text(0.02, 0.02, metrics_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"分类结果可视化已保存 / Classification visualization saved: {save_path}")

    plt.pause(2)  # 展示2秒 / Display for 2 seconds
    plt.close()

def main():
    """
    主函数 / Main function
    """
    print("=" * 80)
    print("医学图像分类网络演示 Medical Image Classification Network Demo")
    print("=" * 80)

    # 配置参数 / Configuration
    config = ClassificationConfig()
    print(f"配置参数 Configuration: {config}")

    # 生成合成数据 / Generate synthetic data
    print("\n生成合成医学图像数据 / Generating synthetic medical image data...")
    images, labels = generate_synthetic_medical_data(config, num_samples=50)

    # 数据预处理 / Data preprocessing
    print("\n执行数据预处理 / Performing data preprocessing...")
    # 转换为PyTorch张量 / Convert to PyTorch tensors
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
    labels_tensor = torch.from_numpy(labels)

    # 划分训练和测试集 / Split train and test sets
    split_idx = int(0.8 * len(images_tensor))
    train_images, test_images = images_tensor[:split_idx], images_tensor[split_idx:]
    train_labels, test_labels = labels_tensor[:split_idx], labels_tensor[split_idx:]

    print(f"训练集大小 Training set size: {len(train_images)}")
    print(f"测试集大小 Test set size: {len(test_images)}")

    # 创建数据加载器 / Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 初始化模型 / Initialize model
    print("\n初始化分类模型 / Initializing classification model...")
    model = MedicalImageClassifier(config).to(config.device)
    print(f"模型参数数量 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟训练过程 / Simulate training process
    print("\n开始模型训练模拟 / Starting model training simulation...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 简单训练循环 / Simple training loop
    for epoch in range(5):  # 简化为5轮训练 / Simplified to 5 epochs
        model.train()
        train_loss = 0.0
        for i in range(0, len(train_images), config.batch_size):
            batch_images = train_images[i:i+config.batch_size].to(config.device)
            batch_labels = train_labels[i:i+config.batch_size].to(config.device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"训练轮次 Epoch {epoch+1}/5, 损失 Loss: {train_loss/len(train_images):.4f}")

    # 评估模型 / Evaluate model
    print("\n评估模型性能 / Evaluating model performance...")
    metrics, all_preds, all_labels = evaluate_classification_metrics(model, test_loader, config.device)

    print(f"评估结果 Evaluation Results:")
    print(f"  准确率 Accuracy: {metrics['accuracy']:.4f}")
    print(f"  精确率 Precision: {metrics['precision']:.4f}")
    print(f"  召回率 Recall: {metrics['recall']:.4f}")
    print(f"  F1分数 F1-Score: {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    # 可视化结果 / Visualize results
    print("\n生成可视化结果 / Generating visualization results...")
    test_images_np = test_images.permute(0, 2, 3, 1).numpy()  # [N, C, H, W] -> [N, H, W, C]
    test_labels_np = test_labels.numpy()

    visualize_classification_results(
        test_images_np, test_labels_np, all_preds, metrics,
        save_path="output/medical_classification_results.png"
    )

    # 保存结果 / Save results
    print("\n保存分类报告 / Saving classification report...")
    results = {
        'configuration': {
            'image_size': config.image_size,
            'num_classes': config.num_classes,
            'device': config.device
        },
        'dataset_info': {
            'total_samples': len(images),
            'train_samples': len(train_images),
            'test_samples': len(test_images),
            'normal_samples': int(np.sum(labels == 0)),
            'abnormal_samples': int(np.sum(labels == 1))
        },
        'performance_metrics': metrics,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }

    with open("output/medical_classification_report.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("医学图像分类网络演示完成 Medical Image Classification Demo Completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()