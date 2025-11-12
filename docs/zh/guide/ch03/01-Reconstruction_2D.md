# 3.1 CT重建

## 1. 引言
计算机断层扫描（CT）重建是指从多个角度采集的X射线投影数据中恢复物体二维（2D）或三维（3D）图像的过程。它将投影数据（正弦图）转换为描述组织内部线性衰减系数的空间域图像。

CT重建在数学上基于**Radon变换**，该变换模拟了衰减函数的线积分如何对应投影测量值。本节介绍几种核心重建方法，包括**滤波反投影（FBP）**、用于锥束几何的**Feldkamp-Davis-Kress（FDK）**算法，以及带有**正则化（Tikhonov，TV）**的**迭代重建（SART，OSEM）**。

---

## 2. 投影几何

### 2.1 平行束几何
在平行束系统中，每个视角的X射线都是平行的，每个探测器元件测量沿其对应射线的总衰减。虽然概念简单，但平行束几何在现代CT扫描仪中很少使用，因为它需要平移而不是旋转。

数学上，投影p(θ, t)可以表示为：

$$p(\theta, t) = \int_{-\infty}^{+\infty} f(x, y) \, \delta(t - x\cos\theta - y\sin\theta) \, dx\,dy$$

其中：
- $f(x, y)$是衰减系数分布
- $\theta$是投影角度
- $t$是探测器坐标
- $\delta$是狄拉克δ函数

### 2.2 扇束几何
扇束与平行束投影之间的关系：

$$p_{\text{fan}}(\beta, \gamma) = p_{\text{parallel}}(\theta = \beta + \gamma, t = R \sin \gamma)$$

其中：
- $\beta$是光源的旋转角度
- $\gamma$是扇角
- $R$是光源到中心的距离

### 2.3 锥束几何
在锥束系统中，光源发射的射线在横向和纵向上都发散，形成3D锥形。这种几何允许在单次旋转中获取体积数据，但使图像重建更加复杂。诸如Feldkamp-Davis-Kress（FDK）之类的算法被用来有效地近似完整的3D重建。

---

## 3. Radon变换与滤波反投影（FBP）

### 3.1 Radon变换

Radon变换描述了投影数据如何从图像生成。对于二维图像f(x, y)，Radon变换Rf(θ, s)表示f沿角度θ和偏移s的线的线积分。

它可以概念性地表示为：

Rf(θ, s) = ∫ f(s·cosθ - t·sinθ, s·sinθ + t·cosθ) dt  (从-∞到+∞)

其中：
- f(x, y)是原始图像
- θ是投影角度
- s是探测器坐标
- Rf(θ, s)是正弦图

**逆Radon变换**从在所有角度范围内收集的投影Rf(θ, s)重建f(x, y)。

---

### 3.2 滤波反投影（FBP）
**滤波反投影（FBP）** 是大多数临床CT扫描仪中使用的经典解析重建算法。它通过应用频域滤波器然后将滤波结果反投影到图像空间来从投影中恢复图像。

FBP过程包括两个主要步骤：

1. **滤波**
   每个投影首先在探测器域中与高通滤波器进行卷积，以校正简单反投影中固有的频率不平衡。
   常用滤波器包括Ram-Lak（斜坡）、Shepp-Logan和Hann滤波器。

2. **反投影**
   滤波后的投影然后沿着它们最初采集的相同路径被反投影回图像域。
   对于每个像素(x, y)，算法对所有投影角度的贡献求和。

以紧凑的数学形式表示，重建可以表示为：

$$f(x, y) = \int_0^{\pi} [Rf(\theta, s) * h(s)] \bigg|_{s = x\cos\theta + y\sin\theta} \, d\theta$$

其中：
- $h(s)$是卷积滤波器核（斜坡滤波器）
- $*$表示卷积
- 积分累积所有投影角度的贡献

FBP在理想条件下提供快速准确的重建，但对噪声、有限的角度覆盖和不一致的测量敏感。

---

### 3.3 实际考虑
- FBP假设完整且均匀采样的投影；欠采样会导致条纹伪影。
- 滤波器的选择直接影响噪声和分辨率。
  - Ram-Lak产生清晰的图像但会放大噪声。
  - Hann或Hamming滤波器提供更平滑、噪声抑制的结果。
- 在低剂量或稀疏视图CT中，FBP性能下降，促使使用迭代重建方法。

---

### 3.4 滤波器比较示例

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# 生成测试图像
image = shepp_logan_phantom()
angles = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=angles, circle=True)

# 使用不同滤波器进行重建
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
reconstructions = []

for filter_name in filters:
    reconstruction = iradon(sinogram, theta=angles, 
                           filter_name=filter_name, circle=True)
    reconstructions.append(reconstruction)

# 显示对比结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('原始图像')

for i, (filter_name, recon) in enumerate(zip(filters, reconstructions)):
    row, col = (i+1) // 3, (i+1) % 3
    axes[row, col].imshow(recon, cmap='gray')
    axes[row, col].set_title(f'{filter_name.title()} 滤波器')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```
## 4. FDK算法（3D锥束重建）

Feldkamp-Davis-Kress（FDK）算法是一种为圆形扫描轨迹采集的3D锥束CT数据设计的近似解析重建方法。它通过引入几何加权来校正锥束发散，扩展了2D滤波反投影（FBP）方法。

FDK算法因其在计算效率和重建精度之间的平衡而广泛应用于锥束CT（CBCT）应用，如牙科成像、介入放射学和工业检测。

---

### 4.1 基本原理

在锥束CT系统中，X射线源沿着圆形路径围绕物体旋转，每个探测器元件收集代表沿每个X射线路径的积分衰减的数据。

对于给定的投影角度β和探测器坐标(u, v)，检测到的投影值p(u, v, β)对应于沿射线的总衰减。目标是从所有这样的投影重建3D体积f(x, y, z)。


---

### 4.2 算法步骤

1. **加权**
   为了校正锥束发散，每个探测器像素乘以一个基于距离的加权因子：

   $$w(u, v) = \frac{D}{\sqrt{D^2 + u^2 + v^2}}$$

   其中D是从X射线源到探测器平面的距离。

2. **滤波**
   每个探测器行（沿u方向）与一维斜坡滤波器h(u)进行卷积，以执行高通频率补偿，类似于二维滤波反投影（FBP）过程。

   $$p_{\text{filtered}}(u, v, \beta) = p_{\text{weighted}}(u, v, \beta) * h(u)$$

   斜坡滤波器增强高频分量，这提高了空间分辨率但也会增加噪声。在实践中，通常应用更平滑的滤波器（如Hann或Hamming）来减少高频放大。

3. **反投影**
   滤波后的数据被反投影到三维重建体积中。
   对于每个体素(x, y, z)，算法计算每个投影角度β对应的探测器坐标(u, v)，并累积来自所有视角的加权投影值。

   重建值f(x, y, z)计算为所有投影角度的积分：

   $$f(x, y, z) = \int_0^{2\pi} \frac{D^2}{(D + x \cos\beta + y \sin\beta)^2} \cdot p_{\text{filtered}}\big( u(x, y, \beta), v(x, y, z, \beta), \beta \big)  d\beta$$

   其中几何映射函数定义为：

   $$u(x, y, \beta) = \frac{D (x \sin\beta - y \cos\beta)}{D + x \cos\beta + y \sin\beta}$$

   $$v(x, y, z, \beta) = \frac{D z}{D + x \cos\beta + y \sin\beta}$$

   这些映射函数描述了3D对象中的每个体素如何对应不同投影角度的探测器位置。在实现过程中，通常需要插值，因为(u, v)坐标可能无法精确对齐离散的探测器像素。

---

### 4.3 实际注意事项

- FDK算法假设锥角相对较小。
  当锥角较大或扫描轨迹非圆形（例如，螺旋CT）时，需要更高级的算法，如Grangeat或Katsevich方法。

- 斜坡滤波器可以使用基于快速傅里叶变换（FFT）的卷积高效实现。

- 在现代系统中，FDK反投影步骤被高度并行化，并通常在图形处理单元（GPU）上实现，以实现近实时重建。

---

### 4.4 示例（伪代码）

以下是一个简化的伪代码，演示FDK算法的基本步骤。

```python
# 简化的FDK重建伪代码

# 给定：投影数据 P(u, v, beta) 和几何参数 D

for each beta in [0, 2*pi]:
    for each (u, v) in detector:
        weight[u, v] = D / sqrt(D**2 + u**2 + v**2)
        P_weighted[u, v, beta] = P[u, v, beta] * weight[u, v]

# 沿u方向滤波（使用斜坡滤波器）
for each beta:
    for each v:
        P_filtered[:, v, beta] = ramp_filter(P_weighted[:, v, beta])

# 反投影
for each voxel (x, y, z):
    f[x, y, z] = 0
    for each beta:
        u = D * (x * sin(beta) - y * cos(beta)) / (D + x * cos(beta) + y * sin(beta))
        v = D * z / (D + x * cos(beta) + y * sin(beta))
        f[x, y, z] += (D**2 / (D + x * cos(beta) + y * sin(beta))**2) * P_filtered[u, v, beta]
```
### 4.5 计算特性

| 属性 | 描述 |
|-----------|--------------|
| **类型** | 解析重建（FBP到3D锥束几何的扩展） |
| **计算复杂度** | `O(N³)`，用于重建 `N × N × N` 体素体积 |
| **内存需求** | 高，与 `N³ + N² × M` 成正比，其中 `M` 是投影数量 |
| **精度** | 对于小锥角精度高，对于大锥角由于圆形轨迹假设而近似 |
| **噪声敏感性** | 与FBP类似——对高频噪声敏感；可以使用变迹滤波器（例如Hann、Hamming）缓解 |
| **并行化** | 高度可并行化；GPU加速（CUDA/OpenCL）常用于临床CBCT重建 |
| **典型用例** | 牙科CBCT、介入CT、微型CT和工业无损检测 |
| **优点** | 快速、确定性且兼容实时应用 |
| **局限性** | 对于宽锥角或非圆形扫描轨迹精度有限 |

## 5. 迭代重建与正则化

解析重建方法如FBP和FDK计算效率高，但依赖于理想假设——密集投影采样、高信噪比（SNR）和准确的系统几何。在实际设置中，尤其是在**低剂量**或**稀疏视图CT**中，这些条件经常被违反，导致伪影和噪声放大。

**迭代重建（IR）** 方法通过将CT重建表述为一个优化问题来解决这些限制，该问题直接模拟系统物理、噪声统计和先验信息。IR方法通过迭代最小化测量投影和重投影估计之间的差异来重建图像。

---

### 5.1 线性系统模型

将成像域离散化为N个像素或体素后，CT重建问题可以表示为线性方程组：

Af = p

其中每个投影测量：

p_i = Σ(a_ij × f_j) + ε_i  (j从1到N)

- A: 系统矩阵 (M×N)
- f: 未知图像向量
- p: 测量投影向量
- ε_i: 测量噪声

该模型构成了迭代重建算法的基础。目标是找到f的估计值，最小化测量数据p和前向投影Af之间的差异。

---

### 5.2 代数重建方法（ART和SART）

**代数重建技术（ART）** 通过顺序校正估计图像来更新图像。
**同步代数重建技术（SART）** 是更稳定的变体，在处理投影子集后同时更新所有像素。

SART更新规则：

f^(k+1) = f^(k) + λ × [A^T (p - A f^(k))] / (Σ a_ij)

参数说明：
- f^(k): 第k次迭代的重建图像
- λ: 松弛参数 (0.5-1.0)
- A: 系统矩阵
- A^T: 矩阵转置
- p: 测量投影向量

SART收敛更快，对噪声数据更鲁棒，适用于稀疏视图或低剂量CT应用。

---

### 5.3 统计重建：OSEM

对于遵循泊松噪声统计的发射断层扫描或低剂量CT数据，**有序子集期望最大化（OSEM）** 被广泛使用。
它将投影数据划分为子集以加速标准期望最大化（EM）算法。

更新规则为：

$$f^{(k+1)} = f^{(k)} \cdot \frac{A^T \left( \frac{p}{A f^{(k)}} \right)}{A^T \mathbf{1}}$$

这种乘法方案确保非负性，并通过每次迭代处理较小的数据子集提供更快的收敛。

---

### 5.4 正则化方法

正则化结合先验信息以稳定重建、抑制噪声并保留图像结构。
两种常见形式是**Tikhonov（L2）** 和**全变分（TV）** 正则化。

#### (a) Tikhonov（L2）正则化

也称为**L2正则化**，它惩罚大的像素值并强制平滑度：

$$\min_f \; \|A f - p\|_2^2 + \lambda \|f\|_2^2$$

小规模问题的解析解为：

$$f = (A^T A + \lambda I)^{-1} A^T p$$

然而，对于大型CT系统，使用迭代求解器（例如共轭梯度）。

#### (b) 全变分（TV）正则化

TV正则化促进分段平滑同时保留边缘：

$$\min_f \; \|A f - p\|_2^2 + \lambda \|\nabla f\|_1$$

这种方法构成了**压缩感知CT**的基础，从稀疏或低剂量数据实现高质量重建。

---

### 5.5 示例：带TV去噪的迭代SART重建

以下Python演示使用`skimage.transform.iradon_sart`执行迭代重建，并使用`skimage.restoration.denoise_tv_chambolle`应用基于TV的去噪。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon_sart
from skimage.restoration import denoise_tv_chambolle

# 创建Shepp-Logan幻影
image = shepp_logan_phantom()
angles = np.linspace(0., 180., 45, endpoint=False)  # 稀疏视图投影
sinogram = radon(image, theta=angles, circle=True)

# 执行迭代SART重建
reconstruction_sart = iradon_sart(sinogram, theta=angles)
for _ in range(9):  # 总共10次迭代
    reconstruction_sart = iradon_sart(sinogram, theta=angles, image=reconstruction_sart)

# 应用全变分（TV）去噪作为正则化
reconstruction_tv = denoise_tv_chambolle(reconstruction_sart, weight=0.05)

# 可视化
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(image, cmap='gray'); ax[0].set_title('原始图像'); ax[0].axis('off')
ax[1].imshow(reconstruction_sart, cmap='gray'); ax[1].set_title('SART重建'); ax[1].axis('off')
ax[2].imshow(reconstruction_tv, cmap='gray'); ax[2].set_title('SART + TV去噪'); ax[2].axis('off')
plt.tight_layout(); plt.show()
```
## 6. CT重建方法比较总结

本节比较上述主要CT重建算法在计算特性、噪声行为和实践应用方面的差异。
每种方法在速度、精度和对不完整或噪声数据的鲁棒性之间提供权衡。

### 6.1 解析方法

**滤波反投影（FBP）** 和**Feldkamp-Davis-Kress（FDK）** 属于解析重建算法类。
它们使用涉及滤波和反投影的数学公式直接反演Radon变换（或其3D扩展）。
它们的关键优势是计算速度，但假设投影数据密集、无噪声且几何准确。

典型特征：

- **FBP**是临床CT中使用的标准2D重建方法。
  它简单、确定，并且可以在毫秒内重建切片。
  但是，它对噪声和角度欠采样敏感。

- **FDK**将FBP扩展到3D锥束几何。
  它高效且广泛用于牙科和介入成像的锥束CT（CBCT），但对于大锥角或非圆形轨迹变得近似。

### 6.2 迭代方法

**迭代重建（IR）**算法，如**SART**和**OSEM**，迭代最小化测量投影和预测投影之间的差异。
它们可以明确包括系统建模、统计噪声和物理先验，从而改善图像质量，特别是对于低剂量或稀疏视图数据。

特征：

- **SART（同步代数重建技术）**
  为有限数据提供稳定收敛和良好结果。
  该方法计算密集但提供高噪声鲁棒性。

- **OSEM（有序子集期望最大化）**
  通过将投影数据拆分为子集来加速收敛。
  常见于发射断层扫描（PET/SPECT）并可适应具有统计噪声模型的CT系统。

### 6.3 正则化和压缩感知方法

正则化引入先验约束以改善重建问题的条件。
常见形式包括**Tikhonov（L2）** 和**全变分（TV）** 正则化。

- **Tikhonov正则化** 添加L2惩罚项以稳定解并减少噪声但可能过度平滑细节。
- **全变分（TV）** 最小化梯度幅度，有效抑制噪声同时保留边缘。
  它构成了许多用于低剂量或有限角度成像的压缩感知CT算法的基础。

### 6.4 总结表

| 方法 | 类型 | 优点 | 局限性 | 典型应用 |
|--------|------|------------|--------------|----------------------|
| **FBP** | 解析（2D） | 极快；易于实现 | 对噪声和稀疏采样敏感 | 常规临床CT |
| **FDK** | 解析（3D） | 高效体积重建 | 对大锥角近似 | 牙科CBCT，介入CT |
| **SART** | 迭代（代数） | 对稀疏或噪声数据鲁棒 | 高计算成本 | 低剂量或少视图CT |
| **OSEM** | 迭代（统计） | 通过子集快速收敛 | 需要统计噪声建模 | PET，SPECT，低剂量CT |
| **Tikhonov** | 正则化（L2） | 平滑噪声；稳定解 | 模糊细节 | 预处理迭代CT |
| **TV** | 正则化（L1梯度） | 保留边缘；抑制噪声 | 优化复杂性 | 压缩感知CT |

### 6.5 实践指南

1. **使用解析方法（FBP，FDK）**:当数据完整且噪声适中时；它们是常规临床成像的标准。
2. **采用迭代方法（SART，OSEM）**:用于低剂量、稀疏或不完整投影数据，其中解析方法产生伪影。
3. **应用正则化（Tikhonov，TV）**:以稳定迭代重建并实现噪声抑制同时保持边缘。
4. **组合方法**：许多现代系统使用混合流水线——解析初始化后跟迭代细化。
5. **考虑计算**：解析方法随数据大小线性扩展，而迭代方法可能需要数十或数百次迭代，强烈受益于GPU加速。

---

## 7. 延伸阅读：深度学习CT重建

近年来，深度学习在CT重建领域取得了显著进展，主要方法包括：

### 7.1 基于深度学习的后处理方法
- **FBPConvNet**: 将传统FBP与CNN结合进行后处理去噪
- **RED-CNN**: 专用于低剂量CT去噪的残差编码器-解码器网络
- **GAN-based方法**: 使用生成对抗网络提升重建图像质量

### 7.2 深度迭代重建方法
- **Learned Primal-Dual**: 将迭代重建算法展开为深度网络
- **ADMM-Net**: 基于交替方向乘子法的可学习重建网络
- **MODL**: 基于字典学习的模型驱动深度网络

### 7.3 端到端深度学习重建
- **DeepPET**: 直接从投影数据重建图像的端到端网络
- **iCT-Net**: 专为稀疏视图CT设计的集成学习网络
- **DuDoNet**: 双域网络，同时在投影域和图像域进行学习

### 7.4 优势与挑战
**优势**:
- 在极低剂量条件下仍能保持良好图像质量
- 重建速度相比传统迭代方法显著提升
- 能够学习复杂的噪声和伪影模式

**挑战**:
- 需要大量高质量训练数据
- 模型泛化能力和可解释性有待提高
- 临床验证和标准化仍需时间

### 7.5 推荐文献
1. **Jin et al.**, "Deep Convolutional Neural Network for Inverse Problems in Imaging", *IEEE TIP*, 2017.
2. **Yang et al.**, "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction", *CVPR*, 2019.
3. **Wang et al.**, "iCT-Net: Integrate CNN and Transformer for Sparse-View CT Reconstruction", *Medical Physics*, 2022.

---

## 参考文献

1. **A. C. Kak** and **M. Slaney**, *Principles of Computerized Tomographic Imaging*, SIAM, 2001.  
   （CT几何、Radon变换和解析重建理论的综合参考。）

2. **L. A. Feldkamp**, **L. C. Davis**, and **J. W. Kress**, "Practical cone-beam algorithm," *Journal of the Optical Society of America A*, vol. 1, no. 6, pp. 612–619, 1984.  
   （描述锥束CT的FDK算法的原始论文。）

3. **A. H. Andersen** and **A. C. Kak**, "Simultaneous Algebraic Reconstruction Technique (SART): A superior implementation of the ART algorithm," *Ultrasonic Imaging*, vol. 6, pp. 81–94, 1984.  
   （介绍SART方法的基础工作。）

4. **H. M. Hudson** and **R. S. Larkin**, "Accelerated image reconstruction using ordered subsets of projection data," *IEEE Transactions on Medical Imaging*, vol. 13, no. 4, pp. 601–609, 1994.  
   （关于OSEM的开创性论文，一种快速迭代统计重建算法。）

5. **L. I. Rudin**, **S. Osher**, and **E. Fatemi**, "Nonlinear total variation based noise removal algorithms," *Physica D: Nonlinear Phenomena*, vol. 60, no. 1–4, pp. 259–268, 1992.  
   （介绍用于边缘保留去噪的全变分最小化。）

6. **E. Y. Sidky** and **X. Pan**, "Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization," *Physics in Medicine and Biology*, vol. 53, no. 17, pp. 4777–4807, 2008.  
   （演示锥束CT的基于TV的迭代重建。）

7. **Jin et al.**, "Deep Convolutional Neural Network for Inverse Problems in Imaging", *IEEE TIP*, 2017.
8. **Yang et al.**, "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction", *CVPR*, 2019.
9. **Wang et al.**, "iCT-Net: Integrate CNN and Transformer for Sparse-View CT Reconstruction", *Medical Physics*, 2022.



