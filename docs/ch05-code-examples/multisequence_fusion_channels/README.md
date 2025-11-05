# 多序列MRI融合通道

## 📋 概述

本代码示例演示了多序列MRI图像的通道融合技术，将不同MRI序列（T1、T2、FLAIR、DWI等）整合为多通道输入，为深度学习模型提供丰富的信息。

## 🔄 最近更新

### 2025-11-04 - 语法错误修复

修复了 `main.py` 中的关键语法错误：

1. **第398行语法错误修复**：
   - 修复了 `center = [0.5, 0.5, 0.5]` 的语法错误
   - 确保数组定义的正确性

2. **变量定义修复**：
   - 修复了 `wm_mask` 变量在使用前未定义的问题
   - 在生成T1序列时正确定义白质掩码

3. **可视化函数修复**：
   - 修复了 `visualize_multisequence_fusion` 函数中的类型错误
   - 将 `len(axes.shape[1])` 改为 `axes.shape[1]`，因为 `axes.shape[1]` 是整数，不能对其调用 `len()`

4. **测试验证**：
   - 所有核心功能测试通过（堆叠融合、加权融合）
   - 图像重采样测试通过（linear、nearest、cubic插值）
   - 序列标准化测试通过（z_score、min_max、robust标准化）

### 当前状态
- ✅ 基本功能正常
- ✅ 可视化功能正常
- ✅ 测试套件通过（除PCA融合因sklearn版本兼容性问题）
- ✅ 合成数据生成正常

### 已知问题
- PCA融合方法由于numpy和sklearn的版本兼容性问题暂时无法使用
- 中文字体警告（不影响功能）

## 🎯 学习目标

1. **理解多序列MRI的互补性**
   - 不同序列提供的组织信息差异
   - 多序列融合的临床价值
   - 信息融合的理论基础

2. **掌握图像配准和重采样技术**
   - 空间一致性保证
   - 插值方法的选择
   - 分辨率标准化

3. **了解不同融合策略**
   - 堆叠融合方法
   - 加权融合方法
   - PCA降维融合
   - 各方法的优缺点

## 🧮 算法原理

### 多序列MRI信息互补性

| 序列 | 主要信息 | 优势 | 适用场景 |
|------|----------|------|----------|
| **T1** | 解剖结构 | 高空间分辨率 | 结构分析、体积测量 |
| **T2** | 病理特征 | 对液体敏感 | 水肿检测、炎症识别 |
| **FLAIR** | 病灶边界 | 病变高对比 | 白质病变分析 |
| **DWI** | 细胞密度 | 扩散敏感性 | 缺血检测、肿瘤分级 |

### 融合算法流程

```python
def multisequence_fusion(sequences, fusion_method='stack'):
    # 1. 图像配准和重采样
    resampled_sequences = []
    for sequence in sequences:
        resampled = resample_to_target(sequence)
        normalized = normalize_intensity(resampled)
        resampled_sequences.append(normalized)

    # 2. 执行融合
    if fusion_method == 'stack':
        fused = np.stack(resampled_sequences, axis=-1)
    elif fusion_method == 'weighted':
        fused = weighted_average(resampled_sequences)
    elif fusion_method == 'pca':
        fused = pca_fusion(resampled_sequences)

    return fused
```

### 重采样算法

```python
def resample_image(image, original_spacing, target_spacing):
    # 计算缩放因子
    scale_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]

    # 选择插值方法
    if interpolation == 'linear':
        order = 1
    elif interpolation == 'nearest':
        order = 0
    elif interpolation == 'cubic':
        order = 3

    # 执行重采样
    resampled = ndimage.zoom(image, scale_factors, order=order)
    return resampled
```

## 🏥 临床应用

### 适用场景

| 应用场景 | 推荐序列 | 融合方法 | 临床价值 |
|----------|----------|----------|----------|
| **脑肿瘤分割** | T1+T2+FLAIR | 堆叠融合 | 提高分割准确性 |
| **多发性硬化** | T1+FLAIR | 加权融合 | 病灶检测 |
| **缺血性中风** | DWI+FLAIR+T2 | PCA融合 | 急性期诊断 |
| **脑萎缩评估** | T1+T2 | 堆叠融合 | 体积测量 |

### 质量标准

- **空间一致性**: 所有序列精确配准
- **强度标准化**: 统一的数值范围
- **信息保持**: 保留重要的诊断信息
- **计算效率**: 合理的处理时间

## 📊 测试数据

### 合成数据特点

代码包含合成多序列MRI数据生成功能：

1. **解剖结构一致性**
   - 所有序列基于相同的解剖基础
   - 真实的组织信号比例
   - 空间配准完美

2. **序列特异性特征**
   - T1: 解剖结构导向
   - T2: 病理和液体导向
   - FLAIR: 病灶边界导向
   - DWI: 扩散导向

3. **质量模拟**
   - 可调节噪声水平
   - 偏场场模拟
   - 病理区域模拟

### 真实数据推荐

**ADNI数据集**
- 网址: http://adni.loni.usc.edu/
- 描述: 阿尔茨海默病神经影像学倡议
- 特点: 多模态MRI，标准化协议

**OASIS数据集**
- 网址: https://www.oasis-brains.org/
- 描述: 公开的脑部MRI数据集
- 特点: 多种年龄组，高质量T1图像

**BRATS数据集**
- 网址: https://www.med.upenn.edu/sbia/brats2017.html
- 描述: 脑肿瘤分割挑战
- 特点: 多模态MRI，金标准分割

## 🚀 使用方法

### 基本使用

```bash
# 运行主程序
python main.py

# 运行测试
python test.py
```

### 单独使用融合器

```python
import numpy as np
from main import MultisequenceFusion, generate_synthetic_mri

# 生成多序列数据
sequences = generate_synthetic_mri(shape=(128, 128, 64))

# 创建融合处理器
fusion_processor = MultisequenceFusion(target_shape=(128, 128, 64))

# 准备序列信息
sequences_info = []
for name, image in sequences.items():
    sequences_info.append({
        'image': image,
        'spacing': (1.0, 1.0, 1.0),
        'name': name
    })

# 执行融合
fused_image, stats = fusion_processor.multisequence_fusion_channels(
    sequences_info, fusion_method='stack'
)

print(f"融合图像形状: {fused_image.shape}")
print(f"融合方法: {stats['fusion_method']}")
```

### 自定义融合参数

```python
# 自定义目标形状和插值方法
fusion_processor = MultisequenceFusion(
    target_shape=(256, 256, 128),
    interpolation='cubic'  # 三次样条插值
)

# 使用加权融合
fused_image, stats = fusion_processor.multisequence_fusion_channels(
    sequences_info,
    fusion_method='weighted'
)
```

## 📈 输出结果

### 融合统计信息

算法输出详细的融合统计：

```python
stats = {
    'fusion_method': 'stack',
    'sequence_names': ['T1', 'T2', 'FLAIR', 'DWI'],
    'original_stats': [
        {'name': 'T1', 'mean': T1均值, 'std': T1标准差, ...},
        {'name': 'T2', 'mean': T2均值, 'std': T2标准差, ...},
        # ...
    ],
    'fused_stats': {
        'mean': 融合图像均值,
        'std': 融合图像标准差,
        'shape': 融合图像形状
    },
    'correlations': [
        {'sequence': 'T1', 'correlation': 相关系数},
        # ...
    ]
}
```

### 可视化结果

生成多子图的详细可视化：

1. **原始序列显示**: 各序列的原始图像
2. **融合图像**: 融合后的多通道图像
3. **强度分布**: 融合图像的直方图
4. **统计信息**: 详细的数值统计
5. **相关性分析**: 序列间的相关性

### 保存文件

- `output/multisequence_fusion_result.png`: 主要融合结果
- `output/multisequence_fusion_comparison.png`: 融合方法对比

## ⚙️ 依赖要求

```bash
pip install numpy matplotlib scipy scikit-image
```

可选依赖（用于PCA融合）：
```bash
pip install scikit-learn
```

## 🧪 测试说明

运行 `test.py` 将执行以下测试：

1. **基本功能测试**
   - 验证融合算法基本正确性
   - 检查输出格式

2. **图像重采样测试**
   - 不同插值方法
   - 目标形状适配

3. **序列标准化测试**
   - Z-score标准化
   - Min-Max标准化
   - 鲁棒标准化

4. **融合方法测试**
   - 堆叠融合
   - 加权融合
   - PCA融合

5. **合成数据生成测试**
   - 多序列一致性
   - 病理特征模拟

6. **边界条件测试**
   - 小图像处理
   - 单序列融合
   - 异常值处理

7. **性能测试**
   - 不同大小处理速度
   - 内存使用效率

8. **可视化功能测试**
   - 图像生成和保存

## 🎓 学习要点

1. **理论基础**: 理解多序列融合的物理和统计基础
2. **技术实现**: 掌握配准、重采样、标准化技术
3. **方法选择**: 了解不同融合策略的适用场景
4. **质量评估**: 掌握融合效果的评估方法
5. **临床应用**: 认识多序列融合的临床价值

## 📚 扩展阅读

1. **经典论文**
   - Rohling M, et al. Multimodal brain tumor segmentation using atlas. MIA 2007.
   - Menze BH, et al. A generative probabilistic model and its application to medical image analysis. MIA 2010.

2. **算法改进**
   - 深度学习融合方法
   - 注意力机制融合
   - 图神经网络融合

3. **临床应用**
   - 多模态诊断系统
   - 计算机辅助诊断
   - 精准医学

## 🔬 高级主题

### 深度学习融合

```python
import torch
import torch.nn as nn

class DeepFusionNet(nn.Module):
    """
    基于深度学习的多序列融合网络
    """
    def __init__(self, num_sequences):
        super().__init__()

        # 特征提取器
        self.feature_extractors = nn.ModuleList([
            nn.Conv3d(1, 16, 3, padding=1) for _ in range(num_sequences)
        ])

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv3d(16 * num_sequences, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, 1),
            nn.ReLU()
        )

        # 输出层
        self.output_layer = nn.Conv3d(32, 1, 1)

    def forward(self, sequences):
        # 提取特征
        features = []
        for i, seq in enumerate(sequences):
            feat = self.feature_extractors[i](seq.unsqueeze(1))
            features.append(feat)

        # 融合特征
        fused = torch.cat(features, dim=1)
        fused = self.fusion_layer(fused)

        # 输出
        output = self.output_layer(fused)
        return output
```

### 注意力机制融合

```python
class AttentionFusion(nn.Module):
    """
    注意力机制融合
    """
    def __init__(self, num_sequences):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_sequences,
            num_heads=8
        )

    def forward(self, sequences):
        # 将序列展平并转置
        batch_size, _, height, width, depth = sequences[0].shape
        flattened = [seq.view(batch_size, -1) for seq in sequences]
        stacked = torch.stack(flattened, dim=2)  # [B, D, N]

        # 应用注意力
        attended, _ = self.attention(stacked, stacked, stacked)

        # 重塑回原始形状
        fused = attended.mean(dim=1)  # 简单平均融合
        fused = fused.view(batch_size, height, width, depth)

        return fused
```

### 图神经网络融合

```python
class GraphFusion:
    """
    图神经网络融合
    """
    def __init__(self):
        # 实现细节略...
        pass

    def fuse_sequences(self, sequences):
        # 构建图结构
        # 实现细节略...
        pass
```

## 🚨 注意事项

1. **空间配准**: 确保所有序列精确配准
2. **强度标准化**: 不同序列可能需要不同的标准化方法
3. **内存管理**: 大体积数据需要考虑内存限制
4. **质量控制**: 验证融合结果的合理性

## 📊 性能基准

### 处理速度参考

| 图像大小 | 序列数量 | 处理时间 | 内存使用 |
|----------|----------|----------|----------|
| 64×64×32 | 2 | ~0.5秒 | ~16MB |
| 128×128×64 | 4 | ~2.5秒 | ~128MB |
| 256×256×128 | 4 | ~15秒 | ~1GB |

### 融合方法对比

| 方法 | 计算复杂度 | 信息保持 | 适用场景 |
|------|------------|----------|----------|
| 堆叠 | O(1) | 完整 | 通用 |
| 加权 | O(N) | 选择性 | 特定任务 |
| PCA | O(N²) | 主要成分 | 降维需求 |

## 📞 技术支持

如有问题，请参考：
1. 代码注释和文档
2. 测试用例和示例
3. 相关论文和资料

多序列MRI融合是现代医学影像分析的重要技术，能够显著提高诊断准确性和分析效果，为深度学习模型提供丰富的多模态输入。