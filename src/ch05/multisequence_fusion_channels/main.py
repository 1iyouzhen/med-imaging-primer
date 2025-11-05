#!/usr/bin/env python3
"""
多序列MRI融合通道算法实现
实现多种MRI序列的通道融合，包括完整的测试和可视化功能

学习目标：
1. 理解多序列MRI的信息互补性
2. 掌握图像配准和重采样技术
3. 了解不同融合策略的优缺点

算法原理：
多序列MRI融合是将不同MRI序列的信息整合为多通道输入的过程：

核心思想：
- 不同序列提供互补的组织信息
- T1: 解剖结构信息
- T2: 病理和液体信息
- FLAIR: 病灶边界信息
- DWI: 细胞密度信息

融合步骤：
1. 图像配准和重采样
2. 强度标准化
3. 通道叠加
4. 质量评估

技术特点：
- 保持空间一致性
- 标准化强度范围
- 保持组织对比度
- 支持不同融合策略
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, measure, transform
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

class MultisequenceFusion:
    """
    多序列MRI融合处理器
    """

    def __init__(self, target_shape=(128, 128, 64), interpolation='linear'):
        """
        初始化融合处理器

        参数:
            target_shape (tuple): 目标形状 (height, width, depth)
            interpolation (str): 插值方法 ('linear', 'nearest', 'cubic')
        """
        self.target_shape = target_shape
        self.interpolation = interpolation

        print(f"多序列融合处理器初始化:")
        print(f"  目标形状: {target_shape}")
        print(f"  插值方法: {interpolation}")

    def resample_image(self, image, original_spacing, target_spacing=None):
        """
        图像重采样

        参数:
            image (numpy.ndarray): 输入图像
            original_spacing (tuple): 原始间距
            target_spacing (tuple): 目标间距

        返回:
            numpy.ndarray: 重采样后的图像
        """
        if target_spacing is None:
            # 计算目标间距以匹配目标形状
            target_spacing = tuple(
                orig * size / target for orig, size, target in
                zip(original_spacing, image.shape, self.target_shape)
            )

        print(f"重采样: {image.shape} -> {self.target_shape}")
        print(f"  原始间距: {original_spacing}")
        print(f"  目标间距: {target_spacing}")

        # 计算缩放因子
        scale_factors = [
            orig / target for orig, target in zip(original_spacing, target_spacing)
        ]

        # 选择插值方法
        if self.interpolation == 'linear':
            order = 1
        elif self.interpolation == 'nearest':
            order = 0
        elif self.interpolation == 'cubic':
            order = 3
        else:
            order = 1

        # 执行重采样
        resampled = ndimage.zoom(image, scale_factors, order=order)

        # 裁剪或填充到目标形状
        if resampled.shape != self.target_shape:
            # 简单的中心裁剪或填充
            resampled = self._resize_to_target(resampled)

        print(f"重采样完成，最终形状: {resampled.shape}")
        return resampled

    def _resize_to_target(self, image):
        """
        调整图像到目标形状
        """
        if len(image.shape) != len(self.target_shape):
            raise ValueError("图像维度与目标不匹配")

        # 对于每个维度，进行中心裁剪或填充
        result = np.zeros(self.target_shape, dtype=image.dtype)

        # 计算裁剪/填充参数
        slices = []
        for dim in range(len(self.target_shape)):
            if image.shape[dim] >= self.target_shape[dim]:
                # 裁剪
                start = (image.shape[dim] - self.target_shape[dim]) // 2
                end = start + self.target_shape[dim]
                slices.append(slice(start, end))
            else:
                # 填充
                pad_before = (self.target_shape[dim] - image.shape[dim]) // 2
                pad_after = self.target_shape[dim] - image.shape[dim] - pad_before
                slices.append(slice(None))

        # 应用裁剪/填充
        try:
            cropped = image[tuple(slices)]
            # 填充到完整形状
            for dim in range(len(self.target_shape)):
                if image.shape[dim] < self.target_shape[dim]:
                    pad_before = (self.target_shape[dim] - image.shape[dim]) // 2
                    pad_after = self.target_shape[dim] - image.shape[dim] - pad_before

                    pad_width = [(0, 0)] * len(self.target_shape)
                    pad_width[dim] = (pad_before, pad_after)

                    cropped = np.pad(cropped, pad_width, mode='constant', constant_values=0)

            result = cropped
        except:
            # 如果裁剪失败，使用简单的缩放
            scale = [target / size for target, size in zip(self.target_shape, image.shape)]
            result = ndimage.zoom(image, scale, order=1)

        return result

    def normalize_sequence(self, image, method='z_score'):
        """
        序列标准化

        参数:
            image (numpy.ndarray): 输入图像
            method (str): 标准化方法 ('z_score', 'min_max', 'robust')

        返回:
            numpy.ndarray: 标准化后的图像
        """
        print(f"序列标准化 (方法: {method})")
        print(f"  输入范围: [{np.min(image):.3f}, {np.max(image):.3f}]")

        if method == 'z_score':
            # Z-score标准化
            normalized = (image - np.mean(image)) / (np.std(image) + 1e-8)
            print(f"  Z-score标准化: 均值={np.mean(normalized):.3f}, 标准差={np.std(normalized):.3f}")

        elif method == 'min_max':
            # 最小-最大标准化
            min_val, max_val = np.min(image), np.max(image)
            normalized = (image - min_val) / (max_val - min_val + 1e-8)
            print(f"  Min-Max标准化: 范围=[{np.min(normalized):.3f}, {np.max(normalized):.3f}]")

        elif method == 'robust':
            # 鲁棒标准化（使用中位数和四分位距）
            median_val = np.median(image)
            q75, q25 = np.percentile(image, [75, 25])
            iqr = q75 - q25
            normalized = (image - median_val) / (iqr + 1e-8)
            print(f"  鲁棒标准化: 中位数={median_val:.3f}, IQR={iqr:.3f}")

        else:
            raise ValueError(f"未知的标准化方法: {method}")

        print(f"  输出范围: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        return normalized

    def multisequence_fusion_channels(self, sequences_info, fusion_method='stack'):
        """
        多序列融合为多通道

        参数:
            sequences_info (list): 序列信息列表，每个元素为{'image': array, 'spacing': tuple, 'name': str}
            fusion_method (str): 融合方法 ('stack', 'weighted', 'pca')

        返回:
            tuple: (融合图像, 融合统计信息)
        """
        print(f"多序列融合开始，方法: {fusion_method}")
        print(f"输入序列数量: {len(sequences_info)}")

        # 步骤1: 重采样所有序列到目标形状和间距
        resampled_sequences = []
        sequence_names = []

        for i, seq_info in enumerate(sequences_info):
            print(f"\n处理序列 {i+1}: {seq_info['name']}")
            print(f"  原始形状: {seq_info['image'].shape}")
            print(f"  原始间距: {seq_info['spacing']}")

            # 重采样
            resampled = self.resample_image(
                seq_info['image'],
                seq_info['spacing']
            )

            # 标准化
            normalized = self.normalize_sequence(resampled, method='z_score')

            resampled_sequences.append(normalized)
            sequence_names.append(seq_info['name'])

        # 步骤2: 执行融合
        if fusion_method == 'stack':
            fused_image = self._stack_fusion(resampled_sequences)
        elif fusion_method == 'weighted':
            fused_image = self._weighted_fusion(resampled_sequences)
        elif fusion_method == 'pca':
            fused_image = self._pca_fusion(resampled_sequences)
        else:
            raise ValueError(f"未知的融合方法: {fusion_method}")

        # 计算统计信息
        stats = self._calculate_fusion_stats(
            resampled_sequences, fused_image, sequence_names, fusion_method
        )

        print(f"\n融合完成!")
        print(f"融合图像形状: {fused_image.shape}")
        print(f"融合方法: {fusion_method}")

        return fused_image, stats

    def _stack_fusion(self, sequences):
        """
        堆叠融合方法
        """
        stacked = np.stack(sequences, axis=-1)  # (H, W, D, C)
        return stacked

    def _weighted_fusion(self, sequences):
        """
        加权融合方法
        """
        # 计算每个序列的权重（基于方差）
        weights = []
        for seq in sequences:
            # 方差越大，权重越高（包含更多信息）
            weight = np.var(seq)
            weights.append(weight)

        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        print(f"融合权重: {weights}")

        # 加权平均
        fused = np.zeros_like(sequences[0])
        for i, (seq, weight) in enumerate(zip(sequences, weights)):
            fused += seq * weight

        # 添加通道维度
        return np.expand_dims(fused, axis=-1)

    def _pca_fusion(self, sequences):
        """
        PCA融合方法
        """
        try:
            import numpy as np
            import sklearn
            # 检查版本兼容性
            numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
            sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
            
            # 检查numpy 2.x与scikit-learn 1.3.x的兼容性问题
            if numpy_version[0] >= 2 and sklearn_version[:2] == (1, 3):
                raise ImportError("numpy 2.x与scikit-learn 1.3.x存在二进制兼容性问题")
                
            from sklearn.decomposition import PCA

            # 将3D图像展平为2D
            flattened_sequences = []
            for seq in sequences:
                flattened = seq.reshape(-1)
                flattened_sequences.append(flattened)

            # 创建数据矩阵
            data_matrix = np.column_stack(flattened_sequences)

            # PCA降维
            pca = PCA(n_components=min(3, len(sequences)))
            principal_components = pca.fit_transform(data_matrix)

            # 选择主要成分
            fused_principal = principal_components[:, 0]  # 第一主成分

            # 重塑为原始形状
            fused = fused_principal.reshape(sequences[0].shape)

            # 添加通道维度
            return np.expand_dims(fused, axis=-1)
            
        except ImportError as e:
            if "numpy.dtype size changed" in str(e) or "二进制兼容性" in str(e):
                error_msg = f"PCA融合失败: numpy {np.__version__}与scikit-learn {sklearn.__version__}版本不兼容"
                print(f"  ❌ {error_msg}")
                print("  建议解决方案: pip install 'numpy<2.0.0' 'scikit-learn<1.4.0'")
                raise ImportError(error_msg)
            else:
                raise

    def _calculate_fusion_stats(self, original_sequences, fused_image, sequence_names, fusion_method):
        """
        计算融合统计信息
        """
        stats = {
            'fusion_method': fusion_method,
            'sequence_names': sequence_names,
            'original_stats': [],
            'fused_stats': {},
            'correlations': []
        }

        # 计算原始序列统计
        for i, (seq, name) in enumerate(zip(original_sequences, sequence_names)):
            seq_stats = {
                'name': name,
                'mean': np.mean(seq),
                'std': np.std(seq),
                'min': np.min(seq),
                'max': np.max(seq),
                'shape': seq.shape
            }
            stats['original_stats'].append(seq_stats)

        # 计算融合图像统计
        stats['fused_stats'] = {
            'mean': np.mean(fused_image),
            'std': np.std(fused_image),
            'min': np.min(fused_image),
            'max': np.max(fused_image),
            'shape': fused_image.shape
        }

        # 计算相关性（如果只有单通道融合）
        if fused_image.shape[-1] == 1:
            fused_flat = fused_image.flatten()
            for i, (seq, name) in enumerate(zip(original_sequences, sequence_names)):
                seq_flat = seq.flatten()
                correlation = np.corrcoef(seq_flat, fused_flat)[0, 1]
                stats['correlations'].append({
                    'sequence': name,
                    'correlation': correlation
                })

        return stats

def generate_synthetic_multisequence_mri(shape=(64, 64, 32), noise_level=0.05, bias_field_strength=0.2):
    """
    生成合成的多序列MRI数据

    参数:
        shape (tuple): 图像形状
        noise_level (float): 噪声水平
        bias_field_strength (float): 偏场场强度

    返回:
        dict: 包含不同序列的数据
    """
    print(f"生成合成多序列MRI数据 (形状: {shape})")

    # 创建坐标网格
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    z, y, x = z / shape[0], y / shape[1], x / shape[2]

    sequences = {}

    # T1序列（解剖结构导向）
    print("生成T1序列...")
    t1_image = np.zeros(shape, dtype=np.float32)
    center = [0.5, 0.5, 0.5]

    # 白质
    wm_mask = ((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2 <= 0.15)
    t1_image[wm_mask] = 0.8

    # 灰质
    gm_mask = ((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2 <= 0.25) & ~wm_mask
    t1_image[gm_mask] = 0.5

    # 脑脊液
    csf_mask = ((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2 <= 0.05)
    t1_image[csf_mask] = 0.1

    # 添加噪声和偏场场
    t1_image += np.random.normal(0, noise_level, shape)
    if bias_field_strength > 0:
        bias_field = 1 + bias_field_strength * 0.2 * np.sin(2 * np.pi * y)
        t1_image *= bias_field

    t1_image = np.clip(t1_image, 0, 1)
    sequences['T1'] = t1_image

    # T2序列（病理导向）
    print("生成T2序列...")
    t2_image = t1_image.copy() * 0.6  # 基于T1结构

    # 增强液体信号
    t2_image[csf_mask] = 0.9  # CSF高信号

    # 添加病理区域（高信号）
    lesion_center = [0.7, 0.3, 0.6]
    lesion_mask = ((x - lesion_center[2])**2 + (y - lesion_center[1])**2 + (z - lesion_center[0])**2) <= 0.02
    t2_image[lesion_mask] = 0.8

    t2_image += np.random.normal(0, noise_level * 1.2, shape)
    t2_image = np.clip(t2_image, 0, 1)
    sequences['T2'] = t2_image

    # FLAIR序列（病灶边界导向）
    print("生成FLAIR序列...")
    flair_image = t1_image.copy() * 0.7

    # 抑制液体信号
    flair_image[csf_mask] = 0.1  # CSF低信号

    # 增强病灶边界
    from scipy import ndimage
    lesion_mask_dilated = ndimage.binary_dilation(lesion_mask, iterations=2)
    flair_image[lesion_mask_dilated & ~csf_mask] = 0.9

    flair_image += np.random.normal(0, noise_level * 0.8, shape)
    flair_image = np.clip(flair_image, 0, 1)
    sequences['FLAIR'] = flair_image

    # DWI序列（扩散导向）
    print("生成DWI序列...")
    dwi_image = t1_image.copy() * 0.4

    # 模拟扩散限制区域
    restriction_center = [0.3, 0.7, 0.4]
    restriction_mask = ((x - restriction_center[2])**2 + (y - restriction_center[1])**2 + (z - restriction_center[0])**2) <= 0.03
    dwi_image[restriction_mask] = 0.1  # 扩散限制区域低信号

    dwi_image += np.random.normal(0, noise_level * 1.5, shape)
    dwi_image = np.clip(dwi_image, 0, 1)
    sequences['DWI'] = dwi_image

    print(f"多序列MRI数据生成完成")
    for name, image in sequences.items():
        print(f"  {name}: 范围=[{np.min(image):.3f}, {np.max(image):.3f}]")

    return sequences

def visualize_multisequence_fusion(sequences, fused_image, stats, save_path=None):
    """
    可视化多序列融合结果

    参数:
        sequences (dict): 原始序列数据
        fused_image (numpy.ndarray): 融合图像
        stats (dict): 融合统计信息
        save_path (str): 保存路径
    """
    print("生成多序列融合可视化...")

    sequence_names = list(sequences.keys())
    num_sequences = len(sequence_names)

    # 选择中间切片显示
    if len(fused_image.shape) == 4:
        mid_slice = fused_image.shape[2] // 2
        fused_slice = fused_image[:, :, mid_slice, :] if fused_image.shape[-1] == 1 else fused_image[:, :, mid_slice, 0]
    else:
        mid_slice = fused_image.shape[2] // 2
        fused_slice = fused_image[:, :, mid_slice]

    # 创建可视化
    fig, axes = plt.subplots(2, max(3, num_sequences), figsize=(4*max(3, num_sequences), 8))

    # 第一行：原始序列
    for i, name in enumerate(sequence_names):
        if i < axes.shape[1]:
            seq_slice = sequences[name][:, :, mid_slice]
            im = axes[0, i].imshow(seq_slice, cmap='gray')
            axes[0, i].set_title(f'{name} 序列')
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046)

    # 第二行：融合结果和分析
    # 融合图像
    im = axes[1, 0].imshow(fused_slice, cmap='gray')
    axes[1, 0].set_title('融合图像')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # 强度分布对比
    axes[1, 1].hist(fused_slice.flatten(), bins=50, alpha=0.7, color='blue', density=True)
    axes[1, 1].set_title('融合图像强度分布')
    axes[1, 1].set_xlabel('强度值')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].grid(True, alpha=0.3)

    # 打印统计信息到控制台（避免图片中的中文乱码问题）
    print("\n融合统计信息:")
    print(f"方法: {stats['fusion_method']}")
    print(f"形状: {stats['fused_stats']['shape']}")
    print(f"均值: {stats['fused_stats']['mean']:.3f}")
    print(f"标准差: {stats['fused_stats']['std']:.3f}")
    print(f"范围: [{stats['fused_stats']['min']:.3f}, {stats['fused_stats']['max']:.3f}]")

    if stats['correlations']:
        print("相关性:")
        for corr in stats['correlations']:
            print(f"  {corr['sequence']}: {corr['correlation']:.3f}")

    # 移除统计信息子图，改为显示其他有用信息
    if axes.shape[1] > 2:
        # 可以显示其他可视化内容，比如不同序列的对比
        axes[1, 2].axis('off')  # 暂时隐藏这个子图

    plt.suptitle(f'多序列MRI融合效果分析\n序列: {", ".join(sequence_names)}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    plt.pause(2)  # 展示2秒
    plt.close()

def compare_fusion_methods(sequences):
    """
    比较不同融合方法
    """
    print("比较不同融合方法...")

    # 创建融合处理器
    fusion_processor = MultisequenceFusion(target_shape=(64, 64, 32))

    # 准备序列信息
    sequences_info = []
    for name, image in sequences.items():
        sequences_info.append({
            'image': image,
            'spacing': (1.0, 1.0, 1.0),  # 假设各向同性
            'name': name
        })

    # 测试不同融合方法
    fusion_methods = ['stack', 'weighted', 'pca']
    results = {}

    for method in fusion_methods:
        print(f"\n测试融合方法: {method}")

        try:
            if method == 'pca':
                # PCA需要sklearn，跳过如果不可用
                try:
                    from sklearn.decomposition import PCA
                    fused_image, stats = fusion_processor.multisequence_fusion_channels(
                        sequences_info, fusion_method=method
                    )
                except ImportError:
                    print("  跳过PCA方法（需要sklearn）")
                    continue
            else:
                fused_image, stats = fusion_processor.multisequence_fusion_channels(
                    sequences_info, fusion_method=method
                )

            results[method] = {
                'fused_image': fused_image,
                'stats': stats
            }

            print(f"  [OK] {method} 融合完成")

        except Exception as e:
            print(f"  [ERROR] {method} 融合失败: {e}")

    # 可视化对比
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 6))

        for i, (method, result) in enumerate(results.items()):
            fused_image = result['fused_image']
            stats = result['stats']

            if len(fused_image.shape) == 4:
                mid_slice = fused_image.shape[2] // 2
                display_image = fused_image[:, :, mid_slice, 0]
            else:
                mid_slice = fused_image.shape[2] // 2
                display_image = fused_image[:, :, mid_slice]

            axes[i].imshow(display_image, cmap='gray')
            axes[i].set_title(f'{method.capitalize()} 融合\n形状: {fused_image.shape}')
            axes[i].axis('off')

        plt.suptitle('多序列融合方法对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        os.makedirs("output", exist_ok=True)
        save_path = "output/multisequence_fusion_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.pause(2)  # 展示2秒
        plt.close()

    return results

def main():
    """
    主函数：演示多序列融合的完整流程
    """
    print("多序列MRI融合演示")
    print("="*50)

    try:
        # 生成合成多序列MRI数据
        print("\n" + "="*50)
        print("生成合成多序列MRI数据")
        print("="*50)

        sequences = generate_synthetic_multisequence_mri(
            shape=(64, 64, 32), noise_level=0.05, bias_field_strength=0.2
        )

        # 创建融合处理器
        fusion_processor = MultisequenceFusion(target_shape=(64, 64, 32))

        # 准备序列信息
        sequences_info = []
        for name, image in sequences.items():
            sequences_info.append({
                'image': image,
                'spacing': (1.0, 1.0, 1.0),
                'name': name
            })

        # 执行融合
        print("\n" + "="*50)
        print("执行多序列融合")
        print("="*50)

        fused_image, fusion_stats = fusion_processor.multisequence_fusion_channels(
            sequences_info, fusion_method='stack'
        )

        # 可视化结果
        print("\n" + "="*50)
        print("生成可视化结果")
        print("="*50)

        os.makedirs("output", exist_ok=True)
        save_path = "output/multisequence_fusion_result.png"

        visualize_multisequence_fusion(
            sequences, fused_image, fusion_stats, save_path
        )

        # 比较不同融合方法
        compare_fusion_methods(sequences)

        print("\n" + "="*60)
        print("总结")
        print("="*60)
        print("1. 多序列融合整合了不同MRI序列的互补信息")
        print("2. 图像配准和重采样确保空间一致性")
        print("3. 强度标准化统一不同序列的数值范围")
        print("4. 不同融合方法适用于不同的应用场景")
        print("5. 融合结果为深度学习提供丰富的多通道输入")
        print("6. 质量评估确保融合的有效性")

    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()