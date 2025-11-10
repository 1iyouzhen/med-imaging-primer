# Medical Image Processing Modules

## Overview / 概述

This directory contains two comprehensive medical image processing modules specifically designed for clinical applications and machine learning training. Both modules implement modality-specific augmentation strategies that preserve anatomical constraints and clinical diagnostic value while expanding training datasets.

### 📁 Module Structure / 模块结构
```
src/ch05/
├── medical_image_augmentation/          # General Medical Image Augmentation
│   ├── README.md                        # This documentation
│   ├── main.py                          # Complete implementation with quality metrics
│   ├── simple_augmentation.py           # Simplified version for education
│   └── output/                          # Generated results and reports
└── medical_segmentation_augmentation/   # Segmentation-Specific Augmentation
    ├── README.md                        # Segmentation-specific documentation
    ├── main.py                          # Complete segmentation augmentation
    ├── simple_demo.py                   # Simplified segmentation demo
    └── output/                          # Segmentation visualization results
```

---

## 🏥 Module 1: Medical Image Augmentation (`medical_image_augmentation/`)

### Overview / 概述

Comprehensive medical image augmentation suite with two distinct implementations designed for different use cases - from educational demonstrations to production-level research.

### 📁 File Structure / 文件结构
```
medical_image_augmentation/
├── main.py                    # Complete implementation with research-grade features
├── simple_augmentation.py     # Educational version for learning and demos
├── README.md                  # This documentation
└── output/                    # Generated results
    ├── medical_image_augmentation_ct_demo.png
    └── augmentation_report.json
```

### 🔄 File Comparison / 文件对比

| Feature / 特性 | `main.py` | `simple_augmentation.py` |
|----------------|-----------|---------------------------|
| **Purpose / 用途** | Research & Production | Educational & Demo |
| **Class Design / 类设计** | Configurable class with dataclass | Simple standalone class |
| **Image Size / 图像尺寸** | 512×512 (higher resolution) | 256×256 (faster processing) |
| **Configuration / 配置** | `AugmentationConfig` dataclass | Fixed parameters |
| **Quality Metrics / 质量评估** | ✅ PSNR/SSIM evaluation | ❌ No metrics |
| **Report Generation / 报告生成** | ✅ JSON report output | ❌ No report |
| **Advanced Features / 高级功能** | ✅ CutMix, multiple elastic deformations | ✅ Basic elastic deformation |
| **Modality Support / 模态支持** | CT, MRI, X-ray | CT, MRI, X-ray |
| **Output Detail / 输出详情** | Detailed console output + evaluation | Simple console output |

### 🔧 Technical Differences / 技术差异

#### **main.py - Research Grade / 研究级**
- **Configuration System**: Uses `AugmentationConfig` dataclass for flexible parameter management
- **High Resolution**: 512×512 images for better detail representation
- **Quality Evaluation**: Complete PSNR/SSIM metrics with detailed analysis
- **Advanced Augmentations**:
  - Multiple elastic deformations (configurable parameters)
  - CutMix augmentation for improved generalization
  - Mask-aware transformations (when masks provided)
- **Report Generation**: JSON export with comprehensive statistics
- **Enhanced Visualization**: Quality metrics integration
- **Detailed Logging**: Step-by-step progress tracking with detailed statistics

#### **simple_augmentation.py - Educational / 教育版**
- **Simple Architecture**: Minimal class design for easy understanding
- **Fast Processing**: 256×256 images for quick demonstrations
- **Core Techniques**: Essential augmentation methods only
- **Basic Elastic Deformation**: Single implementation (α=800, σ=6)
- **No Quality Metrics**: Focus on visual demonstration
- **Quick Setup**: No configuration required
- **Educational Output**: Clear, concise console messages

## Features / 特性

### 🏥 Modality-Specific Augmentation / 模态特定增强
- **CT Images**: Hounsfield Unit (HU) value preservation, artifact simulation
- **MRI Images**: Intensity variation, contrast enhancement, noise patterns
- **X-ray Images**: Geometric transformations, density adjustments

### 🔧 Advanced Techniques / 高级技术
- **Elastic Deformation**: Simulates respiratory motion and tissue elasticity
- **Intensity Transformation**: Multi-device protocol adaptation
- **Noise Addition**: Gaussian, Poisson, and speckle noise simulation
- **CutMix**: Advanced mixing augmentation for improved generalization
- **Partial Occlusion**: Metal artifacts, motion blur, grid artifacts

### 📊 Quality Metrics / 质量评估
- **PSNR (Peak Signal-to-Noise Ratio)**: Quantitative quality assessment
- **SSIM (Structural Similarity Index)**: Perceptual quality measurement
- **Histogram Correlation**: Distribution similarity analysis

---

## 🎯 Module 2: Medical Segmentation Augmentation (`medical_segmentation_augmentation/`)

### Overview / 概述

Specialized augmentation techniques designed specifically for medical image segmentation tasks, with strict anatomical constraints and mask-aware transformations that preserve segmentation accuracy while improving model robustness.

### 🔬 Segmentation-Specific Features / 分割特定功能

#### 🧠 Mask-Aware Transformations / 掩码感知变换
- **Synchronized Deformation**: Image and mask transformations applied simultaneously
- **Boundary Preservation**: Maintains edge accuracy between anatomical structures
- **Label Consistency**: Ensures semantic consistency in augmented masks
- **Topology Preservation**: Maintains connectivity of anatomical structures

#### 🏥 Clinical-Grade Augmentations / 临床级增强
- **Elastic Deformation**: Simulates respiratory motion and tissue elasticity (α=1000, σ=8)
- **Intensity Transformation**: Hounsfield Unit (HU) value preservation for CT imaging
- **Realistic Noise**: Gaussian, Poisson, and Rician noise models for different modalities
- **Medical Artifacts**: Metal artifacts, motion blur, and partial volume effects

#### 🎛️ Anatomical Constraints / 解剖学约束
- **Organ Connectivity**: Preserves spatial relationships between structures
- **Physiological Limits**: Respects realistic motion and deformation boundaries
- **Clinical Validity**: Maintains diagnostically relevant features
- **Modality Specificity**: CT, MRI, and X-ray specific parameter ranges

### 📁 Files / 文件结构
```
medical_segmentation_augmentation/
├── README.md                             # Segmentation-specific documentation
├── main.py                               # Complete implementation with advanced analysis
├── simple_demo.py                        # Educational demonstration
└── output/                               # Generated visualizations
    └── medical_segmentation_augmentation_demo.png
```

### 🔧 Core Functions / 核心功能

#### **Elastic Deformation / 弹性变形**
```python
def elastic_deformation(self, image, mask, alpha=1000, sigma=8):
    """
    Apply elastic deformation to both image and mask simultaneously
    Simulates respiratory motion and tissue elasticity
    Parameters:
    - alpha: Deformation strength (default: 1000)
    - sigma: Smoothness control (default: 8)
    """
```

#### **Intensity Transformation / 强度变换**
```python
def intensity_transform(self, image, mask, contrast_factor=1.2, brightness_shift=0):
    """
    Apply HU-aware intensity transformations for CT images
    Preserves clinically relevant intensity ranges
    """
```

#### **Noise Addition / 噪声添加**
```python
def add_noise(self, image, mask, noise_type='gaussian', noise_level=20):
    """
    Add modality-specific noise to simulate acquisition variations
    Supports: Gaussian, Poisson, Rician noise models
    """
```

#### **Partial Occlusion / 部分遮挡**
```python
def add_partial_occlusion(self, image, mask, occlusion_type='metal', severity=0.3):
    """
    Simulate medical artifacts and occlusions
    Types: Metal artifacts, surgical tools, motion artifacts
    """
```

### 📊 Advanced Analysis / 高级分析功能

#### **Dice Coefficient Analysis / Dice系数分析**
- Pre/post-augmentation segmentation quality assessment
- Boundary-preserving transformation validation
- Clinical feature preservation metrics

#### **Structural Similarity / 结构相似性**
- Anatomical structure consistency measurement
- Edge preservation quality assessment
- Spatial relationship maintenance analysis

#### **Clinical Validation Metrics / 临床验证指标**
- Pathological feature preservation verification
- Diagnostic information retention assessment
- Radiological interpretation quality check

### 🖥️ Usage Examples / 使用示例

#### **Quick Demo / 快速演示**
```bash
# Run simple educational demo
python simple_demo.py

# Run complete analysis
python main.py
```

#### **Python Integration / Python集成**
```python
from main import MedicalSegmentationAugmentation

# Initialize augmentor
augmentor = MedicalSegmentationAugmentation(seed=42)

# Create sample medical image (CT lung field)
image, mask = augmentor.create_sample_medical_image()

# Apply segmentation-specific augmentations
augmentations = augmentor.apply_segmentation_augmentation(image, mask)

# Visualize results
augmentor.visualize_augmentation_results(image, mask, augmentations)

# Analyze effects on segmentation quality
analysis = augmentor.analyze_augmentation_effects(image, mask, augmentations)
```

### 📈 Output Visualization / 输出可视化

#### **8-Panel Layout / 8面板布局**
```
[Original] [Original] [Elastic]    [Intensity]
[Image]    [Mask]     [Deformation] [Transform]
[Noise]    [Occlusion] [Analysis]   [Quality]
[Addition] [Simulation] [Results]   [Metrics]
```

#### **Quality Metrics Display / 质量指标显示**
- Dice coefficient preservation
- Boundary edge quality
- Structural similarity scores
- Clinical feature retention rates

---

## 🚀 Installation & Setup / 安装与设置

### Prerequisites / 前置要求
```bash
# Core scientific computing
pip install numpy scipy matplotlib

# Image processing
pip install scikit-image

# Advanced features (optional)
pip install opencv-python  # For additional augmentation techniques
```

### Medical Imaging Specific Libraries / 医学影像专用库
```bash
# For DICOM support (if working with real medical data)
pip install pydicom

# For advanced medical image processing
pip install SimpleITK
pip install nibabel  # For NIfTI format support
```

---

## 🆚 Module Comparison / 模块对比

### When to Use Each Module / 何时使用各模块

| Feature / 特性 | `medical_image_augmentation/` | `medical_segmentation_augmentation/` |
|----------------|------------------------------|-----------------------------------|
| **Target Use / 目标用途** | General ML training | Segmentation-specific tasks |
| **Mask Support / 掩码支持** | No | Yes (image+mask synchronized) |
| **Quality Metrics / 质量评估** | PSNR/SSIM/Histogram | Dice coefficient + PSNR/SSIM |
| **Clinical Validation / 临床验证** | Basic constraints | Strict anatomical constraints |
| **Output Format / 输出格式** | 15-panel visualization | 8-panel analysis layout |
| **Complexity / 复杂度** | Simple to use | Advanced analysis features |

### Use Case Scenarios / 使用场景

#### 🎯 Use `medical_image_augmentation/` when:
- Training classification or detection models
- Need fast, general-purpose augmentation
- Working with large datasets without segmentation masks
- Educational purposes and demonstrations
- Basic research and prototyping

#### 🎯 Use `medical_segmentation_augmentation/` when:
- Training segmentation models (U-Net, DeepLab, etc.)
- Need mask-aware transformations
- Clinical deployment requiring strict validation
- Research on anatomical preservation
- Quality-critical medical applications

---

## 🚀 Usage / 使用方法

### Quick Start / 快速开始

#### General Augmentation / 通用增强

```bash
# Module 1: Medical Image Augmentation
cd medical_image_augmentation

# Simple educational version (fast, 256x256, basic features)
python simple_augmentation.py

# Complete research version (512x512, quality metrics, detailed reports)
python main.py
```

#### **Output Comparison / 输出对比**

**simple_augmentation.py Output:**
```
============================================================
通用医学图像增强演示
============================================================
创建示例医学图像...
选择CT图像进行演示
图像尺寸: (256, 256)
像素值范围: [-1000.0, 1000.0]

应用基础增强技术...
应用强度增强技术...
应用高级增强技术...

生成增强效果可视化...

============================================================
Medical Image Augmentation Statistics:
============================================================
Modality Type: CT
Basic Augmentation: 7 techniques
Intensity Augmentation: 8 techniques
Advanced Augmentation: 3 techniques
Total Techniques: 18

Visualization saved: output/medical_image_augmentation_ct_demo.png
```

**main.py Output:**
```
================================================================================
通用医学图像增强完整演示 / Complete Medical Image Augmentation Demo
================================================================================

[Medical] 创建不同模态的示例医学图像 / Creating sample medical images...
[Select] 选择CT图像进行详细演示 / Selected CT image for detailed demonstration...
图像尺寸 / Image size: (512, 512)
像素值范围 / Pixel range: [-1000.0, 1000.0]

[Basic] 应用基础增强技术 / Applying basic augmentation techniques...
生成 10 种基础增强效果 / Generated 10 basic augmentation effects

[Process] 应用强度增强技术 / Applying intensity augmentation techniques...
生成 11 种强度增强效果 / Generated 11 intensity augmentation effects

[Advanced] 应用高级增强技术 / Applying advanced augmentation techniques...
生成 6 种高级增强效果 / Generated 6 advanced augmentation effects

[Visualize] 生成增强效果可视化 / Generating augmentation visualization...

================================================================================
Medical Image Augmentation - Quality Evaluation:
================================================================================
Modality Type: CT
Image Size: (512, 512)
Pixel Range: [-1000.0, 1000.0]
Basic Augmentation: 10 techniques
Intensity Augmentation: 10 techniques
Advanced Augmentation: 6 techniques
Total Techniques: 27

Quality Metrics Summary:
  Average PSNR: 8.33 dB
  Average SSIM: 0.940

Visualization saved: output/medical_image_augmentation_ct_demo.png
```

#### Segmentation Augmentation / 分割增强
```bash
# Module 2: Segmentation-specific augmentation
cd medical_segmentation_augmentation

# Simple demonstration
python simple_demo.py

# Complete analysis with quality metrics
python main.py
```

### Advanced Integration / 高级集成

#### Combined Pipeline / 组合流水线
```python
# Step 1: General augmentation for classification
from medical_image_augmentation.simple_augmentation import SimpleMedicalAugmentation
general_aug = SimpleMedicalAugmentation()

# Step 2: Segmentation-specific augmentation
from medical_segmentation_augmentation.main import MedicalSegmentationAugmentation
seg_aug = MedicalSegmentationAugmentation()

# Combined processing
def comprehensive_augmentation(image, mask=None):
    if mask is None:
        # General augmentation for classification
        return general_aug.augment_image(image)
    else:
        # Segmentation-specific augmentation
        return seg_aug.apply_segmentation_augmentation(image, mask)
```

---

## 📊 Output Files / 输出文件

### General Augmentation Module / 通用增强模块
```
medical_image_augmentation/output/
├── medical_image_augmentation_ct_demo.png    # 15-panel visualization
└── augmentation_report.json                   # Statistical report
```

### Segmentation Augmentation Module / 分割增强模块
```
medical_segmentation_augmentation/output/
└── medical_segmentation_augmentation_demo.png  # 8-panel analysis layout
```

---

## 🏥 Clinical Applications / 临床应用

### Training Data Augmentation / 训练数据增强
- **Deep Learning**: Expand datasets for CNN, U-Net, Transformer models
- **Rare Conditions**: Synthesize examples of uncommon pathologies
- **Protocol Harmonization**: Standardize images from different scanners

### Research Applications / 研究应用
- **Algorithm Robustness**: Test model invariance to acquisition variations
- **Validation Studies**: Create controlled test datasets
- **Education**: Demonstrate augmentation effects for teaching

### Quality Control / 质量控制
- **Pipeline Validation**: Verify augmentation doesn't introduce artifacts
- **Consistency Checking**: Ensure multi-modality alignment
- **Clinical Validation**: Expert review of augmented images

---

## 📈 Performance Metrics / 性能指标

### General Augmentation / 通用增强

#### **main.py (Research Version) / 研究版本**
```
Quality Metrics Summary:
  Average PSNR: 8.33 dB
  Average SSIM: 0.940
Total Techniques: 27 (Basic: 10, Intensity: 10, Advanced: 6)
Image Resolution: 512×512
Output Files: PNG visualization + JSON report
```

#### **simple_augmentation.py (Educational Version) / 教育版本**
```
Augmentation Statistics:
Total Techniques: 18 (Basic: 7, Intensity: 8, Advanced: 3)
Image Resolution: 256×256
Output Files: PNG visualization only
Processing Speed: Fast (educational focus)
```

### Segmentation Augmentation / 分割增强
```
Augmentation Analysis Results:
  - Dice Coefficient Preservation
  - Boundary Edge Quality
  - Structural Similarity Scores
  - Clinical Feature Retention Rates
```

---

## 🔧 Troubleshooting / 故障排除

### Common Issues / 常见问题

#### Memory Issues / 内存问题
```python
# Reduce image size or use batch processing
config = AugmentationConfig(image_size=(256, 256))
```

#### Font Issues / 字体问题
- Use English-only labels to avoid font rendering issues
- All outputs are designed to work with standard system fonts

#### Performance Optimization / 性能优化
- Use matplotlib backend: `matplotlib.use('Agg')`
- Implement batch processing for large datasets
- Cache augmentation results for repeated experiments

---

## 🤝 Contributing / 贡献

### Development Guidelines / 开发指南
1. Preserve anatomical constraints in all augmentations
2. Validate with clinical experts when possible
3. Maintain bilingual documentation (Chinese/English)
4. Include quality metrics for all new techniques

### Testing / 测试
```bash
# Test general augmentation
cd medical_image_augmentation && python simple_augmentation.py
cd medical_image_augmentation && python main.py

# Test segmentation augmentation
cd medical_segmentation_augmentation && python simple_demo.py
cd medical_segmentation_augmentation && python main.py
```

### 🎯 Quick Reference / 快速参考

#### **Choose `simple_augmentation.py` when:**
- ✅ Learning medical image augmentation concepts
- ✅ Quick demonstrations and prototyping
- ✅ Educational purposes and teaching
- ✅ Limited computational resources
- ✅ Need fast processing (256×256 images)

#### **Choose `main.py` when:**
- ✅ Research experiments and publications
- ✅ Quality metrics and evaluation needed
- ✅ High-resolution images required (512×512)
- ✅ Detailed reporting and documentation
- ✅ Advanced augmentation techniques (CutMix, multiple elastic deformations)

#### **Expected Results / 预期结果:**
- **simple_augmentation.py**: 18 total techniques, fast execution
- **main.py**: 27 total techniques, PSNR/SSIM evaluation, JSON report

Both files generate the same visualization format and use identical output filenames for consistency.

### Advanced Usage / 高级用法

#### Custom Augmentation Pipeline / 自定义增强流水线
```python
# Create sample images
original_images, masks = augmentor.create_sample_images()

# Apply specific augmentations
ct_image = original_images['CT']

# Basic augmentations
basic_results = augmentor.basic_augmentation(ct_image, 'CT')

# Intensity augmentations
intensity_results = augmentor.intensity_augmentation(ct_image, 'CT')

# Advanced augmentations with mask support
mask = masks['CT']
advanced_results = augmentor.advanced_augmentation(ct_image, mask)

# Evaluate quality
for aug_name, aug_image in basic_results.items():
    metrics = augmentor.evaluate_augmentation(ct_image, aug_image, method='all')
    print(f"{aug_name}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.3f}")
```

## Medical Constraints / 医学约束

### Anatomical Preservation / 解剖学保持
- ✅ Maintains organ relationships and connectivity
- ✅ Preserves tissue boundaries and interfaces
- ✅ Respects physiological motion limits

### Clinical Relevance / 临床相关性
- ✅ Simulates realistic acquisition variations
- ✅ Preserves pathological features
- ✅ Maintains diagnostic value

### Modality-Specific Rules / 模态特定规则
- **CT**: Preserves HU value ranges and attenuation patterns
- **MRI**: Maintains tissue contrast characteristics
- **X-ray**: Respects projection geometry and density relationships

## Output Files / 输出文件

### Visualization Images / 可视化图像
- **Simple Version**: `medical_augmentation_{modality}_demo.png`
  - 3×6 grid layout (18 panels)
  - Shows basic, intensity, and advanced augmentations
  - Chinese labels with detailed descriptions

- **Complete Version**: `medical_image_augmentation_{modality}_demo.png`
  - 3×5 grid layout (15 panels)
  - Enhanced with quality metrics
  - Bilingual labels (Chinese/English)

### Statistical Reports / 统计报告
- **JSON Report**: `augmentation_report.json`
  ```json
  {
    "timestamp": "2025-11-10",
    "modality": "CT",
    "statistics": {
      "basic_augmentation_count": 21,
      "intensity_augmentation_count": 11,
      "advanced_augmentation_count": 6,
      "total_augmentation_count": 27
    },
    "techniques_applied": {
      "basic": ["rotation", "translation", "scale", "flip", ...],
      "advanced": ["elastic_deformation", "cutmix", "occlusion"]
    }
  }
  ```

## Configuration / 配置

### Augmentation Parameters / 增强参数

#### Basic Transformations / 基础变换
```python
# CT specific limits
CT_ROTATION_RANGE = [-5, 5]  # degrees
CT_TRANSLATION_RANGE = 0.05  # fraction of image size
CT_SCALE_RANGE = [0.9, 1.1]  # scaling factor

# MRI specific limits
MRI_ROTATION_RANGE = [-3, 3]
MRI_TRANSLATION_RANGE = 0.03
MRI_SCALE_RANGE = [0.95, 1.05]

# X-ray specific limits
XRAY_ROTATION_RANGE = [-2, 2]
XRAY_TRANSLATION_RANGE = 0.02
XRAY_SCALE_RANGE = [0.98, 1.02]
```

#### Advanced Parameters / 高级参数
```python
# Elastic deformation
ELASTIC_ALPHA = 800  # deformation strength
ELASTIC_SIGMA = 6    # smoothness

# CutMix parameters
CUTMIX_PROBABILITY = 0.5
CUTMIX_BETA = 1.0    # Beta distribution parameter

# Occlusion parameters
OCCLUSION_SEVERITY = 0.3  # 0-1, fraction of image to occlude
```

## Quality Assessment / 质量评估

### Metrics Explanation / 指标说明

#### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 20-50 dB (higher is better)
- **Interpretation**:
  - >30dB: High quality
  - 20-30dB: Moderate quality
  - <20dB: Low quality

#### SSIM (Structural Similarity Index)
- **Range**: 0-1 (higher is better)
- **Interpretation**:
  - >0.9: Excellent structural preservation
  - 0.7-0.9: Good preservation
  - 0.5-0.7: Moderate preservation
  - <0.5: Poor preservation

### Typical Results / 典型结果
```
Rotation / 旋转:
  PSNR: 1.21 dB
  SSIM: 0.874

Translation / 平移:
  PSNR: 1.21 dB
  SSIM: 0.914

Scale / 缩放:
  PSNR: 20.43 dB
  SSIM: 0.994

Flip / 翻转:
  PSNR: 10.46 dB
  SSIM: 0.980
```

## Clinical Applications / 临床应用

### Training Data Augmentation / 训练数据增强
- **Deep Learning**: Expand datasets for CNN, U-Net, Transformer models
- **Rare Conditions**: Synthesize examples of uncommon pathologies
- **Protocol Harmonization**: Standardize images from different scanners

### Research Applications / 研究应用
- **Algorithm Robustness**: Test model invariance to acquisition variations
- **Validation Studies**: Create controlled test datasets
- **Education**: Demonstrate augmentation effects for teaching

### Quality Control / 质量控制
- **Pipeline Validation**: Verify augmentation doesn't introduce artifacts
- **Consistency Checking**: Ensure multi-modality alignment
- **Clinical Validation**: Expert review of augmented images

## Troubleshooting / 故障排除

### Common Issues / 常见问题

#### Unicode Encoding Errors / Unicode编码错误
```bash
# Solution: Set environment variable
export PYTHONIOENCODING=utf-8
# Or use the fixed version with [OK] labels
```

#### Memory Issues / 内存问题
```python
# Reduce image size or batch processing
config = AugmentationConfig(image_size=(256, 256))
```

#### SSIM Calculation Issues / SSIM计算问题
```python
# Ensure images are properly normalized
normalized = augmentor._normalize_for_ssim(image)
```

### Performance Optimization / 性能优化
- Use `matplotlib.use('Agg')` for headless operation
- Implement batch processing for large datasets
- Cache augmentation results for repeated experiments

## Contributing / 贡献

### Development Guidelines / 开发指南
1. Preserve anatomical constraints in all augmentations
2. Validate with clinical experts when possible
3. Maintain bilingual documentation (Chinese/English)
4. Include quality metrics for all new techniques

### Testing / 测试
```bash
# Run basic tests
python simple_augmentation.py

# Run complete tests with metrics
python main.py

# Validate specific modalities
python -c "from main import MedicalImageAugmentation; MedicalImageAugmentation().main()"
```

## License / 许可证

This module is part of the Medical Imaging Primer project. Please refer to the main project license for usage terms.

## Citation / 引用

If you use this module in your research, please cite:

```
Medical Image Augmentation Module (2025)
Medical Imaging Primer - Chapter 5
https://github.com/datawhalechina/med-imaging-primer
```

---

## Technical Notes / 技术说明

### Implementation Details / 实现细节

#### Elastic Deformation Algorithm / 弹性变形算法
- Uses Gaussian-smoothed random displacement fields
- Preserves topology and connectivity
- Simulates physiological motion patterns

#### CutMix Implementation / CutMix实现
- Beta distribution for mixing ratio
- Preserves label consistency
- Supports mask-aware mixing

#### Noise Simulation / 噪声模拟
- **Gaussian**: Electronic noise simulation
- **Poisson**: Photon counting noise (CT specific)
- **Speckle**: Coherent imaging artifacts (ultrasound-like)

### Future Extensions / 未来扩展

#### Planned Features / 计划功能
- [ ] 3D augmentation support
- [ ] DICOM metadata preservation
- [ ] Multi-modal synthesis
- [ ] GAN-based augmentation
- [ ] Clinical validation framework

#### Research Directions / 研究方向
- Physically-based artifact simulation
- Adaptive augmentation based on clinical tasks
- Quality-aware augmentation selection
- Domain-specific constraint learning