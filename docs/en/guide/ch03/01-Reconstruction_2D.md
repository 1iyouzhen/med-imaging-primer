# 3.1 CT Reconstruction 

## 1. Introduction  
Computed Tomography (CT) reconstruction refers to the process of recovering a two-dimensional (2D) or three-dimensional (3D) image of an object from its X-ray projections acquired at multiple angles. It transforms the projection data (sinogram) into spatial domain images that describe the internal linear attenuation coefficients of tissues.

CT reconstruction is mathematically based on the **Radon transform**, which models how line integrals of the attenuation function correspond to projection measurements. This section introduces several core reconstruction approaches, including **Filtered Backprojection (FBP)**, **Feldkamp-Davis-Kress (FDK)** for cone-beam geometry, and **Iterative Reconstruction (SART, OSEM)** with **Regularization (Tikhonov, TV)**.

---

## 2. Projection Geometry

### 2.1 Parallel-beam geometry
In a parallel-beam system, X-rays are parallel for each view, and each detector element measures the total attenuation along its corresponding ray. Although conceptually simple, parallel-beam geometry is rarely used in modern CT scanners because it requires translation instead of rotation.

Mathematically, the projection p(θ, t) can be expressed as :


$$p(\theta, t) = \int_{-\infty}^{+\infty} f(x, y) \, \delta(t - x\cos\theta - y\sin\theta) \, dx\,dy$$

where:
- $f(x, y)$ is the attenuation coefficient distribution
- $\theta$ is the projection angle
- $t$ is the detector coordinate
- $\delta$ is the Dirac delta function

### 2.2 Fan-beam geometry
The relationship between fan-beam and parallel-beam projections:

$$p_{\text{fan}}(\beta, \gamma) = p_{\text{parallel}}(\theta = \beta + \gamma, t = R \sin \gamma)$$

where:
- $\beta$ is the rotation angle of the source
- $\gamma$ is the fan angle  
- $R$ is the source-to-center distance

### 2.3 Cone-beam geometry
In cone-beam systems, the source emits rays diverging both horizontally and vertically, forming a 3D cone shape. This geometry allows volumetric data acquisition in a single rotation but makes image reconstruction more complex. Algorithms such as the Feldkamp–Davis–Kress (FDK) method are used to approximate the full 3D reconstruction efficiently.




---

## 3. Radon Transform and Filtered Backprojection (FBP)

### 3.1 Radon Transform
The Radon transform describes how projection data are generated from an image.  
For a two-dimensional image f(x, y), the Radon transform Rf(θ, s) represents the line integral of f along a line at angle θ and offset s.

It can be expressed conceptually as:


$$Rf(\theta, s) = \int_{-\infty}^{+\infty} f(s \cos\theta - t \sin\theta, s \sin\theta + t \cos\theta) \, dt$$

where:
- $f(x, y)$ is the original image
- $\theta$ is the projection angle
- $s$ is the detector coordinate
- $Rf(\theta, s)$ is the sinogram

The **inverse Radon transform** reconstructs f(x, y) from all its projections Rf(θ, s) collected over a full angular range.

---

### 3.2 Filtered Backprojection (FBP)
**Filtered Backprojection (FBP)** is the classical analytical reconstruction algorithm used in most clinical CT scanners.  
It recovers an image from its projections by applying a frequency-domain filter and then backprojecting the filtered results into image space.

The FBP process consists of two main steps:

1. **Filtering**  
   Each projection is first convolved with a high-pass filter in the detector domain to correct the frequency imbalance inherent in simple backprojection.  
   Common filters include the Ram-Lak (ramp), Shepp–Logan, and Hann filters.

2. **Backprojection**  
   The filtered projections are then smeared back into the image domain along the same paths they were originally acquired from.  
   For every pixel (x, y), the algorithm sums contributions from all projection angles.

In compact mathematical form, the reconstruction can be represented as:


$$f(x, y) = \int_0^{\pi} [Rf(\theta, s) * h(s)] \bigg|_{s = x\cos\theta + y\sin\theta} \, d\theta$$

where:
- $h(s)$ is the convolution filter kernel (ramp filter)
- $*$ denotes convolution
- The integral accumulates contributions from all projection angles

FBP provides fast and accurate reconstruction under ideal conditions, but it is sensitive to noise, limited angular coverage, and inconsistent measurements.

---

### 3.3 Practical Considerations
- FBP assumes complete and uniformly sampled projections; undersampling causes streak artifacts.  
- The choice of filter directly affects noise and resolution.  
  - Ram-Lak yields sharp images but amplifies noise.  
  - Hann or Hamming filters provide smoother, noise-suppressed results.  
- In low-dose or sparse-view CT, FBP performance degrades, motivating the use of iterative reconstruction methods.

---


### 3.4 Filter Comparison Example

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# Generate test image
image = shepp_logan_phantom()
angles = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=angles, circle=True)

# Reconstruction using different filters
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
reconstructions = []

for filter_name in filters:
    reconstruction = iradon(sinogram, theta=angles, 
                           filter_name=filter_name, circle=True)
    reconstructions.append(reconstruction)

# Display comparison results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')

for i, (filter_name, recon) in enumerate(zip(filters, reconstructions)):
    row, col = (i+1) // 3, (i+1) % 3
    axes[row, col].imshow(recon, cmap='gray')
    axes[row, col].set_title(f'{filter_name.title()} Filter')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```
## 4. FDK Algorithm (3D Cone-beam Reconstruction)

The Feldkamp–Davis–Kress (FDK) algorithm is an approximate analytical reconstruction method designed for 3D cone-beam CT data acquired with a circular scanning trajectory. It extends the 2D Filtered Backprojection (FBP) method by introducing geometric weighting to correct for cone-beam divergence.

The FDK algorithm is widely used in cone-beam CT (CBCT) applications such as dental imaging, interventional radiology, and industrial inspection due to its balance between computational efficiency and reconstruction accuracy.

---

### 4.1 Basic Principle

In a cone-beam CT system, the X-ray source rotates around the object along a circular path, and each detector element collects data representing the integrated attenuation along each X-ray path.

For a given projection angle beta, and detector coordinates (u, v), the detected projection value p(u, v, beta) corresponds to the total attenuation along the ray. The goal is to reconstruct the 3D volume f(x, y, z) from all such projections.

---

### 4.2 Algorithm Steps

1. **Weighting**

   To correct for cone-beam divergence, each detector pixel is multiplied by a distance-based weighting factor:

   $$w(u, v) = \frac{D}{\sqrt{D^2 + u^2 + v^2}}$$

   where $D$ is the distance from the X-ray source to the detector plane.

2. **Filtering**

   Each detector row (along the $u$-direction) is convolved with a one-dimensional ramp filter $h(u)$ to perform high-pass frequency compensation, similar to the two-dimensional Filtered Backprojection (FBP) process.

   $$p_{\text{filtered}}(u, v, \beta) = p_{\text{weighted}}(u, v, \beta) * h(u)$$

   The ramp filter enhances high-frequency components, which improves spatial resolution but can also increase noise. In practice, smoother filters such as Hann or Hamming are often applied to reduce high-frequency amplification.

3. **Backprojection**

   The filtered data are then backprojected into the three-dimensional reconstruction volume.  
   For each voxel $(x, y, z)$, the algorithm computes its corresponding detector coordinates $(u, v)$ for each projection angle $\beta$, and accumulates the weighted projection values from all viewing angles.

   Conceptually, the reconstructed value $f(x, y, z)$ is calculated as the sum or integral over all projection angles:

   $$f(x, y, z) = \int_0^{2\pi} \frac{D^2}{(D + x \cos\beta + y \sin\beta)^2} \cdot p_{\text{filtered}}\big( u(x, y, \beta), v(x, y, z, \beta), \beta \big)  d\beta$$

   where the geometric mapping functions are defined as:

   $$u(x, y, \beta) = \frac{D (x \sin\beta - y \cos\beta)}{D + x \cos\beta + y \sin\beta}$$

   $$v(x, y, z, \beta) = \frac{D z}{D + x \cos\beta + y \sin\beta}$$

   These mapping functions describe how each voxel in the 3D object corresponds to detector positions for different projection angles. During implementation, interpolation is often required because $(u, v)$ coordinates may not align exactly with discrete detector pixels.

---

### 4.3 Practical Notes

- The FDK algorithm assumes that the cone angle is relatively small.  
  When the cone angle is large or the scanning trajectory is non-circular (e.g., helical CT), more advanced algorithms such as the Grangeat or Katsevich methods are required.

- The ramp filter can be efficiently implemented using Fast Fourier Transform (FFT)-based convolution.

- In modern systems, the FDK backprojection step is heavily parallelized and often implemented on Graphics Processing Units (GPUs) for near real-time reconstruction.

---

### 4.4 Example (Pseudo-code)

Below is a simplified pseudo-code that demonstrates the essential steps of the FDK algorithm.

```python
# Simplified FDK reconstruction pseudo-code

# Given: projection data P(u, v, beta) and geometry parameter D

for each beta in [0, 2*pi]:
    for each (u, v) in detector:
        weight[u, v] = D / sqrt(D**2 + u**2 + v**2)
        P_weighted[u, v, beta] = P[u, v, beta] * weight[u, v]

# Filtering along u (using ramp filter)
for each beta:
    for each v:
        P_filtered[:, v, beta] = ramp_filter(P_weighted[:, v, beta])

# Backprojection
for each voxel (x, y, z):
    f[x, y, z] = 0
    for each beta:
        u = D * (x * sin(beta) - y * cos(beta)) / (D + x * cos(beta) + y * sin(beta))
        v = D * z / (D + x * cos(beta) + y * sin(beta))
        f[x, y, z] += (D**2 / (D + x * cos(beta) + y * sin(beta))**2) * P_filtered[u, v, beta]

```


### 4.5 Computational Characteristics

| Property | Description |
|-----------|--------------|
| **Type** | Analytic reconstruction (extension of FBP to 3D cone-beam geometry) |
| **Computational complexity** | $O(N^3)$ for reconstructing an $N \times N \times N$ voxel volume |
| **Memory requirement** | High, proportional to $N^3 + N^2 \times M$ where $M$ is number of projections |
| **Accuracy** | High for small cone angles, approximate for large cone angles due to circular trajectory assumption |
| **Noise sensitivity** | Similar to FBP — sensitive to high-frequency noise; can be mitigated using apodized filters (e.g., Hann, Hamming) |
| **Parallelization** | Highly parallelizable; GPU acceleration (CUDA/OpenCL) is commonly used in clinical CBCT reconstruction |
| **Typical use cases** | Dental CBCT, interventional CT, micro-CT, and industrial non-destructive testing |
| **Strengths** | Fast, deterministic, and compatible with real-time applications |
| **Limitations** | Limited accuracy for wide cone angles or non-circular scanning trajectories |

## 5. Iterative Reconstruction and Regularization

Analytical reconstruction methods such as FBP and FDK are computationally efficient but rely on ideal assumptions — dense projection sampling, high signal-to-noise ratio (SNR), and accurate system geometry. In practical settings, especially in **low-dose** or **sparse-view CT**, these conditions are often violated, leading to artifacts and noise amplification.

**Iterative reconstruction (IR)** methods address these limitations by formulating CT reconstruction as an optimization problem that directly models the system physics, noise statistics, and prior information. IR methods reconstruct the image by iteratively minimizing the difference between measured projections and re-projected estimates.

---

### 5.1 Linear System Model

After discretizing the imaging domain into N pixels (for 2D) or voxels (for 3D), the CT reconstruction problem can be represented as a system of linear equations:

$$A f = p$$

where each projection measurement:

$$p_i = \sum_{j=1}^N a_{ij} f_j + \varepsilon_i$$

- $A$: system matrix ($M \times N$)
- $f$: unknown image vector
- $p$: measured projection vector  
- $\varepsilon_i$: measurement noise

Each measured projection value can be viewed as a weighted sum of the attenuation coefficients along a specific X-ray path:

p_i = Σ (from j = 1 to N) [ a_ij * f_j ] + ε_i

where ε_i represents measurement noise (typically Gaussian or Poisson, depending on the acquisition process).

This model forms the foundation for iterative reconstruction algorithms.  
The goal is to find an estimate of **f** that minimizes the difference between the measured data **p** and the forward projection **A * f**, often using optimization-based or statistical methods.
---

### 5.2 Algebraic Reconstruction Methods (ART and SART)

The **Algebraic Reconstruction Technique (ART)** updates the estimated image by sequentially correcting it along each projection direction.  
A more stable and widely used variant is the **Simultaneous Algebraic Reconstruction Technique (SART)**, which updates all pixels simultaneously after processing each subset of projections.

The general SART update rule can be written as:

$$f^{(k+1)} = f^{(k)} + \lambda \cdot \frac{A^T (p - A f^{(k)})}{\sum_j a_{ij}}$$

where:
- $f^{(k)}$ is the reconstructed image at iteration $k$
- $\lambda$ is the relaxation parameter ($0.5 \leq \lambda \leq 1.0$)
- $A$ is the system matrix
- $A^T$ is its transpose
- $p$ is the measured projection vector

SART converges faster and is less sensitive to inconsistent or noisy data than the original ART method.  
It provides a good balance between computational cost and reconstruction quality, making it suitable for sparse-view or low-dose CT applications.

---

### 5.3 Statistical Reconstruction: OSEM

For emission tomography or low-dose CT data that follow Poisson noise statistics, **Ordered Subset Expectation Maximization (OSEM)** [4] is widely used.  
It partitions projection data into subsets to accelerate the standard Expectation–Maximization (EM) algorithm.

The update rule is:

$$f^{(k+1)} = f^{(k)} \cdot \frac{A^T \left( \frac{p}{A f^{(k)}} \right)}{A^T \mathbf{1}}$$


This multiplicative scheme ensures non-negativity and provides faster convergence by processing smaller data subsets per iteration.

---

### 5.4 Regularization Methods

Regularization incorporates prior information to stabilize reconstruction, suppress noise, and preserve image structure.  
Two common forms are **Tikhonov (L2)** and **Total Variation (TV)** regularization.

#### (a) Tikhonov (L2) Regularization

$$\min_f \|A f - p\|_2^2 + \lambda \|f\|_2^2$$

Analytical solution for small-scale problems:

$$f = (A^T A + \lambda I)^{-1} A^T p$$

#### (b) Total Variation (TV) Regularization

$$\min_f \|A f - p\|_2^2 + \lambda \|\nabla f\|_1$$

where $\nabla f$ denotes the image gradient magnitude.


This approach forms the foundation of **compressed-sensing CT**, achieving high-quality reconstructions from sparse or low-dose data [6].

---

### 5.5 Example: Iterative SART Reconstruction with TV Denoising

The following Python demonstration uses `skimage.transform.iradon_sart` to perform iterative reconstruction and `skimage.restoration.denoise_tv_chambolle` to apply TV-based denoising.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon_sart
from skimage.restoration import denoise_tv_chambolle

# Create a Shepp–Logan phantom
image = shepp_logan_phantom()
angles = np.linspace(0., 180., 45, endpoint=False)  # sparse-view projection
sinogram = radon(image, theta=angles, circle=True)

# Perform iterative SART reconstruction
reconstruction_sart = iradon_sart(sinogram, theta=angles)
for _ in range(9):  # total 10 iterations
    reconstruction_sart = iradon_sart(sinogram, theta=angles, image=reconstruction_sart)

# Apply Total Variation (TV) denoising as regularization
reconstruction_tv = denoise_tv_chambolle(reconstruction_sart, weight=0.05)

# Visualization
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(image, cmap='gray'); ax[0].set_title('Original Image'); ax[0].axis('off')
ax[1].imshow(reconstruction_sart, cmap='gray'); ax[1].set_title('SART Reconstruction'); ax[1].axis('off')
ax[2].imshow(reconstruction_tv, cmap='gray'); ax[2].set_title('SART + TV Denoising'); ax[2].axis('off')
plt.tight_layout(); plt.show()
```

## 6. Comparative Summary of CT Reconstruction Methods

This section compares the main CT reconstruction algorithms introduced above in terms of computational characteristics, noise behavior, and practical applications.  
Each method offers a trade-off between speed, accuracy, and robustness to incomplete or noisy data.

### 6.1 Analytic Methods

**Filtered Backprojection (FBP)** and **Feldkamp–Davis–Kress (FDK)** belong to the class of analytic reconstruction algorithms.  
They directly invert the Radon transform (or its 3D extension) using mathematical formulas that involve filtering and backprojection.  
Their key advantage is computational speed, but they assume dense, noise-free, and geometrically accurate projection data.

Typical features:

- **FBP** is the standard 2D reconstruction method used in clinical CT.  
  It is simple, deterministic, and can reconstruct a slice in milliseconds.  
  However, it is sensitive to noise and angular undersampling.

- **FDK** extends FBP to 3D cone-beam geometry.  
  It is efficient and widely used in cone-beam CT (CBCT) for dental and interventional imaging, but becomes approximate for large cone angles or non-circular trajectories.

### 6.2 Iterative Methods

**Iterative Reconstruction (IR)** algorithms, such as **SART** and **OSEM**, iteratively minimize the difference between the measured and predicted projections.  
They can explicitly include system modeling, statistical noise, and physical priors, leading to improved image quality, particularly for low-dose or sparse-view data.

Characteristics:

- **SART (Simultaneous Algebraic Reconstruction Technique)**  
  Provides stable convergence and good results for limited data.  
  The method is computationally intensive but offers high noise robustness.

- **OSEM (Ordered Subset Expectation Maximization)**  
  Accelerates convergence by splitting projection data into subsets.  
  Common in emission tomography (PET/SPECT) and adaptable to CT systems with statistical noise models.

### 6.3 Regularized and Compressed-Sensing Methods

Regularization introduces prior constraints to improve the conditioning of the reconstruction problem.  
Common forms include **Tikhonov (L2)** and **Total Variation (TV)** regularization.

- **Tikhonov regularization** adds an L2 penalty term to stabilize the solution and reduce noise but may oversmooth fine details.  
- **Total Variation (TV)** minimizes the gradient magnitude, effectively suppressing noise while preserving edges.  
  It underlies many compressed-sensing CT algorithms used for low-dose or limited-angle imaging.

### 6.4 Summary Table

| Method | Type | Strengths | Limitations | Typical Applications |
|--------|------|------------|--------------|----------------------|
| **FBP** | Analytic (2D) | Extremely fast; easy to implement | Sensitive to noise and sparse sampling | Conventional clinical CT |
| **FDK** | Analytic (3D) | Efficient volumetric reconstruction | Approximate for large cone angles | Dental CBCT, interventional CT |
| **SART** | Iterative (algebraic) | Robust to sparse or noisy data | High computational cost | Low-dose or few-view CT |
| **OSEM** | Iterative (statistical) | Fast convergence via subsets | Requires statistical noise modeling | PET, SPECT, low-dose CT |
| **Tikhonov** | Regularized (L2) | Smooths noise; stable solution | Blurs fine details | Pre-conditioned iterative CT |
| **TV** | Regularized (L1 gradient) | Preserves edges; suppresses noise | Optimization complexity | Compressed-sensing CT |

### 6.5 Practical Guidelines

1. **Use analytic methods (FBP, FDK)** when data are complete and noise is moderate; they are the standard for routine clinical imaging.  
2. **Adopt iterative methods (SART, OSEM)** for low-dose, sparse, or incomplete projection data where analytic methods produce artifacts.  
3. **Apply regularization (Tikhonov, TV)** to stabilize iterative reconstructions and achieve noise suppression while maintaining edges.  
4. **Combine approaches**: many modern systems use hybrid pipelines—analytic initialization followed by iterative refinement.  
5. **Consider computation**: analytic methods scale linearly with data size, while iterative ones may require tens or hundreds of iterations, benefiting strongly from GPU acceleration.

---

## 7. Extended Reading: Deep Learning CT Reconstruction

Recent years have witnessed significant advances in deep learning for CT reconstruction, with major approaches including:

### 7.1 Deep Learning-Based Post-processing Methods
- **FBPConvNet**: Combines traditional FBP with CNN for post-processing denoising
- **RED-CNN**: Residual encoder-decoder network specialized for low-dose CT denoising
- **GAN-based Methods**: Uses generative adversarial networks to enhance reconstructed image quality

### 7.2 Deep Iterative Reconstruction Methods
- **Learned Primal-Dual**: Unfolds iterative reconstruction algorithms into deep networks
- **ADMM-Net**: Learnable reconstruction network based on alternating direction method of multipliers
- **MODL**: Model-driven deep network based on dictionary learning

### 7.3 End-to-End Deep Learning Reconstruction
- **DeepPET**: End-to-end network for direct image reconstruction from projection data
- **iCT-Net**: Integrated learning network designed for sparse-view CT
- **DuDoNet**: Dual-domain network that learns simultaneously in both projection and image domains

### 7.4 Advantages and Challenges
**Advantages**:
- Maintains good image quality even under extremely low-dose conditions
- Significantly faster reconstruction compared to traditional iterative methods
- Capable of learning complex noise and artifact patterns

**Challenges**:
- Requires large amounts of high-quality training data
- Model generalization and interpretability need improvement
- Clinical validation and standardization still require time

### 7.5 Recommended Literature
1. **Jin et al.**, "Deep Convolutional Neural Network for Inverse Problems in Imaging", *IEEE TIP*, 2017.
2. **Yang et al.**, "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction", *CVPR*, 2019.
3. **Wang et al.**, "iCT-Net: Integrate CNN and Transformer for Sparse-View CT Reconstruction", *Medical Physics*, 2022.

---

## References

1. **A. C. Kak** and **M. Slaney**, *Principles of Computerized Tomographic Imaging*, SIAM, 2001.  
   (Comprehensive reference for CT geometry, Radon transform, and analytic reconstruction theory.)

2. **L. A. Feldkamp**, **L. C. Davis**, and **J. W. Kress**, “Practical cone-beam algorithm,” *Journal of the Optical Society of America A*, vol. 1, no. 6, pp. 612–619, 1984.  
   (Original paper describing the FDK algorithm for cone-beam CT.)

3. **A. H. Andersen** and **A. C. Kak**, “Simultaneous Algebraic Reconstruction Technique (SART): A superior implementation of the ART algorithm,” *Ultrasonic Imaging*, vol. 6, pp. 81–94, 1984.  
   (Foundational work introducing the SART method.)

4. **H. M. Hudson** and **R. S. Larkin**, “Accelerated image reconstruction using ordered subsets of projection data,” *IEEE Transactions on Medical Imaging*, vol. 13, no. 4, pp. 601–609, 1994.  
   (Seminal paper on OSEM, a fast iterative statistical reconstruction algorithm.)

5. **L. I. Rudin**, **S. Osher**, and **E. Fatemi**, “Nonlinear total variation based noise removal algorithms,” *Physica D: Nonlinear Phenomena*, vol. 60, no. 1–4, pp. 259–268, 1992.  
   (Introduces Total Variation minimization for edge-preserving denoising.)

6. **E. Y. Sidky** and **X. Pan**, “Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization,” *Physics in Medicine and Biology*, vol. 53, no. 17, pp. 4777–4807, 2008.  
   (Demonstrates TV-based iterative reconstruction for cone-beam CT.)
7. **Jin et al.**, "Deep Convolutional Neural Network for Inverse Problems in Imaging", *IEEE TIP*, 2017.
8. **Yang et al.**, "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction", *CVPR*, 2019.
9. **Wang et al.**, "iCT-Net: Integrate CNN and Transformer for Sparse-View CT Reconstruction", *Medical Physics*, 2022.



