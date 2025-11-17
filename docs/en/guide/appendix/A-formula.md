# A Key Formulas

> title: A Key Formulas
>
> description: Summarizes the most common, important, and cross-modality universal formulas in medical imaging. Enables readers to quickly consult and understand mathematical derivations when reading the main tutorial. Suitable for beginners to quickly establish a formula framework.

# A Key Formulas

# 1. Introduction

Medical imaging is a typical interdisciplinary field spanning physics, mathematics, engineering, and medicine. Whether it's CT projection reconstruction, MRI Fourier encoding, or PET statistical modeling, the core of these imaging techniques is built upon a series of clear, interpretable, and verifiable mathematical formulas. Understanding these formulas not only helps readers better grasp the principles of image generation but also lays a solid foundation for further study of image reconstruction, image processing, deep learning modeling, and other subsequent content.

This chapter aims to summarize the most commonly used, critical, and cross-modality universal mathematical formulas in medical imaging, organized along the path of "from basic concepts → modality-specific imaging models → image processing → deep learning". Readers can treat this chapter as a quick reference for on-demand consultation while studying the main tutorial, or as an introductory index for further research on imaging physics and reconstruction algorithms.

To ensure coherence, this chapter adopts a concise symbol system and unified expression methods, providing explanations of physical meaning when necessary. Even without a strong mathematical background, readers can establish an understanding of the mathematical framework of medical imaging through this chapter and apply it to practical image analysis and algorithm development.

# 2. Universal Mathematical Foundations in Medical Imaging

This section introduces cross-modality universal mathematical principles in medical imaging, including coordinate systems, sampling, convolution, and noise modeling. These concepts play key roles in the modeling and reconstruction of CT, MRI, X-ray, PET, and ultrasound.

## 2.1 Coordinate Systems and Image Geometry

Geometric relationships in medical imaging primarily involve the mapping of **pixel/voxel coordinates → physical world coordinates**, as well as common affine transformations (rotation, scaling, translation, etc.).

** (1) Pixel/Voxel Coordinates → Physical Space Coordinates**

Imaging files (such as DICOM, NIfTI) typically provide a 4×4 spatial transformation matrix to convert voxel coordinates $(i, j, k)$ to physical coordinates $(x, y, z)$:

<div class="math-display">
\[
\begin{bmatrix}
x \\ y \\ z \\ 1
\end{bmatrix}
=
\mathbf{M}_{\text{DICOM}}
\begin{bmatrix}
i \\ j \\ k \\ 1
\end{bmatrix}
\]
</div>

- $\mathbf{M}_{\text{DICOM}}$ contains:

  - Voxel spacing
  - Image orientation
  - Image origin

---

** (2) DICOM Spatial Transformation Matrix (Typical Form)**

<div class="math-display">
\[
\mathbf{M}_{\text{DICOM}}=
\begin{bmatrix}
\mathbf{R} \cdot \text{diag}(\Delta x,\Delta y,\Delta z) & \mathbf{T} \\
0 & 1
\end{bmatrix}
\]
</div>

Where:

- $\mathbf{R}$: Direction cosine matrix (3×3, specifying the direction of image axes in the world coordinate system)
- $\Delta x, \Delta y, \Delta z$: Voxel size (in mm)
- $\mathbf{T}$: Origin coordinates (in mm in patient coordinate system)

---

** (3) Common Geometric Transformation Matrices**

● Scaling

<div class="math-display">
\[
\mathbf{S} =
\begin{bmatrix}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & s_z
\end{bmatrix}
\]
</div>

● Rotation (using z-axis as example)

<div class="math-display">
\[
\mathbf{R}_z(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta & 0\\
\sin\theta & \cos\theta  & 0\\
0 & 0 & 1
\end{bmatrix}
\]
</div>

● Affine Transform

<div class="math-display">
\[
\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{b}
\]
</div>

Expressed in homogeneous coordinates as:

<div class="math-display">
\[
\begin{bmatrix}
\mathbf{x}' \\ 1
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{A} & \mathbf{b}\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\mathbf{x} \\ 1
\end{bmatrix}
\]
</div>

Affine transformations are widely used in image registration, resampling, and multimodal alignment.

## 2.2 Image Sampling and Convolution

---

** (1) Convolution Definition (Continuous/Discrete)**

● Continuous Convolution

<div class="math-display">
\[
(f * g)(t)=\int_{-\infty}^{+\infty} f(\tau)\, g(t-\tau)\, d\tau
\]
</div>

● Discrete Convolution (Signal)

<div class="math-display">
\[
(f * g)[n] = \sum_{k=-\infty}^{+\infty} f[k]\, g[n-k]
\]
</div>

● 2D Image Convolution (Commonly Used for Filtering)

<div class="math-display">
\[
I'(x,y)=\sum_m \sum_n I(x-m,y-n) \, K(m,n)
\]
</div>

Convolution is frequently used in CT filtered backprojection (FBP), MRI denoising, image smoothing, and sharpening.

---

** (2) Downsampling and Upsampling**

● Downsampling

<div class="math-display">
\[
x_{\text{down}}[n] = x[kN]
\]
</div>

When N = 2, it represents halving of width and height.

● Upsampling

Insert zeros then filter:

<div class="math-display">
\[
x_{\text{up}}[n] =
\begin{cases}
x[n/N], & n \mod N = 0 \\
0, & \text{otherwise}
\end{cases}
\]
</div>

---

** (3) Common Interpolation Formulas**

● Bilinear Interpolation (2D)

<div class="math-display">
\[
f(x,y)=\sum_{m=0}^1\sum_{n=0}^1
f(i+m, j+n)(1-|x-i-m|)(1-|y-j-n|)
\]
</div>

● Trilinear Interpolation (3D)

Used for volumetric data (CT/MRI) resampling, as a 3D extension of bilinear interpolation.

---

** (4) Spatial Domain ↔ Frequency Domain Relationship (Fourier Transform Basics)**

● Fourier Transform (Continuous)

<div class="math-display">
\[
F(\omega)=\int f(t)e^{-j\omega t}dt
\]
</div>

● Discrete Fourier Transform (DFT)

<div class="math-display">
\[
X[k]=\sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}
\]
</div>

● Convolution Theorem (Extremely Important in Medical Imaging)

<div class="math-display">
\[
\mathcal{F}\{f*g\} = \mathcal{F}\{f\}\cdot \mathcal{F}\{g\}
\]
</div>

Explanation:  
**Convolution in the frequency domain is equivalent to multiplication**, which is the core of CT filtered backprojection (FBP) and MRI reconstruction.

## 2.3 Noise Models

Common noise in medical imaging mainly comes from detectors, electronic noise, photon statistical noise, etc.

---

** (1) Additive Gaussian Noise (Common in MRI / CT)**

<div class="math-display">
\[
y = x + \mathcal{N}(0,\sigma^2)
\]
</div>

Applicable scenarios:

- Post-reconstruction CT images
- MRI real/imaginary signal components

---

** (2) Poisson Noise (PET / X-ray)**

PET and X-ray belong to **photon counting processes**, naturally satisfying the Poisson model:

<div class="math-display">
\[
y \sim \text{Poisson}(x)
\]
</div>

Applications:

- PET projection data (sinogram)
- X-ray intensity acquisition
- Noise modeling of CT projection data

Poisson noise is particularly significant in low-dose CT and low-count PET.

# 3. Key Formulas for CT (Computed Tomography)

This section summarizes the most core mathematical formulas in CT imaging and reconstruction, including X-ray attenuation models, Radon transform, filtered backprojection (FBP), as well as CT image HU conversion and windowing processing. These formulas form the mathematical foundation of modern CT imaging technology.

---

## 3.1 X-ray Attenuation Model

When X-rays pass through tissue, they are absorbed and scattered. The attenuation process is described by the **Lambert-Beer Law**, which is the starting point of the CT physical model.

---

** (1) Lambert-Beer Law**

When an X-ray beam passes through a medium, the relationship between the intensity $I$ received by the detector and the incident intensity $I_0$ is:

<div class="math-display">
\[
I = I_0 \exp\left( -\int_L \mu(s)\, ds \right)
\]
</div>

Where:

- $\mu(s)$: Linear attenuation coefficient
- $L$: Photon path
- $I_0$: Incident photon intensity
- $I$: Intensity received by the detector

---

** (2) Projection Data (Log-transform)**

CT projections typically undergo logarithmic transformation to linearize the exponential attenuation:

<div class="math-display">
\[
p = -\ln \left( \frac{I}{I_0} \right)
\]
</div>

This yields:

<div class="math-display">
\[
p = \int_L \mu(s)\, ds
\]
</div>

This is the physical origin of the Radon transform.

---

## 3.2 Radon Transform

The Radon transform describes the line integrals of an object at different angles and is the mathematical representation of CT projection data (sinogram).

---

** (1) Radon Transform Definition**

The projection of a 2D object $\mu(x,y)$ at angle $\theta$ is written as:

<div class="math-display">
\[
p(\theta, t) =
\int_{-\infty}^{+\infty}
\mu(x, y)\, ds
\]
</div>

After parameterizing the integration path:

<div class="math-display">
\[
p(\theta, t)
=
\int_{-\infty}^{+\infty}
\mu(t\cos\theta - s\sin\theta, \;
    t\sin\theta + s\cos\theta)
\, ds
\]
</div>

Where:

- $\theta$: Ray direction
- $t$: Perpendicular distance from the ray to the origin
- $p(\theta,t)$: Sinogram data

---

** (2) Radon Inverse Transform (Ideal Case)**

Theoretically, the object can be recovered through the Radon inverse transform:

<div class="math-display">
\[
\mu(x,y)=\mathcal{R}^{-1}\{p(\theta,t)\}
\]
</div>

Actual reconstruction must combine filtered backprojection (FBP).

---

## 3.3 Filtered Backprojection (FBP)

FBP (Filtered Back Projection) is the classic CT reconstruction algorithm and one of the reconstruction frameworks commonly used in clinical practice today.

---

**(1) Convolution Filtering (Filtering Step)**

Filter the projection $p(\theta,t)$ at each angle:

<div class="math-display">
\[
\tilde{p}(\theta,t)=
p(\theta,t) * h(t)
\]
</div>

Where:

- $h(t)$: Reconstruction filter, for example:

  - **Ram-Lak** (ideal high-pass)
  - Shepp-Logan
  - Hanning, Cosine, etc.

Frequency domain expression of Ram-Lak filter:

<div class="math-display">
\[
H(\omega)=|\omega|
\]
</div>

The purpose of filtering is to compensate for the low-frequency enhancement caused by backprojection.

---

**(2) Backprojection**

Backproject all filtered projections at all angles into the image space:

<div class="math-display">
\[
\mu(x,y)
=
\int_0^{\pi}
\tilde{p}(\theta,  \; x\cos\theta + y\sin\theta )
\; d\theta
\]
</div>

Meaning:

- "Pull back" the 1D filtered projections to their 2D positions
- Accumulate all angles to recover the object's attenuation coefficient distribution

---

**(3) FBP Overall Formula**

<div class="math-display">
\[
\mu(x,y)
=
\int_0^{\pi}
\left[
p(\theta,t) * h(t)
\right]_{t=x\cos\theta+y\sin\theta}
\, d\theta
\]
</div>

---

## 3.4 CT Value (HU) Conversion Formula

CT images are usually measured in HU (Hounsfield Unit), used to quantify the tissue's X-ray attenuation capability.

---

**(1) HU Standardization Formula**

<div class="math-display">
\[
\text{HU} = 1000 \cdot
\frac{\mu - \mu_{\text{water}}}{\mu_{\text{water}}}
\]
</div>

- $\mu$: Tissue attenuation coefficient
- $\mu_{\text{water}}$: Water attenuation coefficient (reference)
- Water = 0 HU
- Air = -1000 HU

---

**(2) Linear Window Function (Windowing)**

Windowing is required when displaying CT images:

<div class="math-display">
\[
I_{\text{display}}
=
\text{clip}\left(
\frac{HU - (WL - \frac{WW}{2})}{WW}
,\, 0,\, 1
\right)
\]
</div>

Where:

- $WW$: Window Width
- $WL$: Window Level
- clip(·): Restricts the result to [0,1]

Examples:

- Soft tissue window: WL=40, WW=400
- Lung window: WL=-600, WW=1500
- Bone window: WL=400, WW=2000

# 4. Key Formulas for MRI (Magnetic Resonance Imaging)

The mathematical foundation of MRI (Magnetic Resonance Imaging) mainly includes the generation of nuclear magnetic resonance signals (Larmor precession and Bloch equations), tissue relaxation processes (T1, T2), and k-space Fourier encoding and reconstruction. This section presents the most core formulas in MRI imaging.

---

## 4.1 Pulse Sequence Basics

MRI signals are determined by the behavior of nuclear magnetic moments under the action of magnetic fields and radio frequency pulses (RF pulse). The basic dynamics are described by Larmor frequency and Bloch equations.

---

**(1) Larmor Frequency Formula**

The precession angular frequency of nuclear spins in a static magnetic field $B_0$ is:

<div class="math-display">
\[
\omega_0 = \gamma B_0
\]
</div>

Where:

- $\omega_0$: Larmor frequency
- $\gamma$: Gyromagnetic ratio (proton: $\gamma/2\pi \approx 42.58\text{ MHz/T}$)
- $B_0$: Main magnetic field strength

Note: High field strength (such as 3T) brings higher signal-to-noise ratio (SNR).

---

**(2) Bloch Equation (Simplified Form)**

Describes the dynamic changes of the magnetization vector $\mathbf{M} = (M_x, M_y, M_z)$ in a magnetic field:

<div class="math-display">
\[
\frac{d\mathbf{M}}{dt}
=
\gamma (\mathbf{M} \times \mathbf{B})
-\frac{M_x \hat{i} + M_y \hat{j}}{T_2}
-\frac{(M_z - M_0)\hat{k}}{T_1}
\]
</div>

Where:

- First term: Precession
- Second term: Transverse relaxation T2
- Third term: Longitudinal relaxation T1
- $M_0$: Equilibrium magnetization

---

## 4.2 T1/T2 Relaxation Signal Models

The process of tissue returning to equilibrium after RF pulse is described by two types of relaxation:

- **T1: Longitudinal recovery**
- **T2: Transverse decay**

---

**(1) T1 Recovery**

<div class="math-display">
\[
M_z(t) = M_0 \left(1 - e^{-t/T_1}\right)
\]
</div>

Meaning:

- $M_z(t)$: Longitudinal magnetization recovering over time
- $T_1$: Longitudinal relaxation time
- White matter, gray matter, and cerebrospinal fluid have different T1 values, used for contrast enhancement

---

**(2) T2 Decay**

<div class="math-display">
\[
M_{xy}(t) = M_0 e^{-t/T_2}
\]
</div>

Note:

- T2 reflects the decay speed of transverse magnetization
- Cerebrospinal fluid has long T2 → bright signal
- White matter has short T2 → dark signal

---

**(3) Proton Density (PD) Signal Expression**

When not strongly dependent on T1/T2, the signal is mainly determined by proton density (PD):

<div class="math-display">
\[
S_{\text{PD}} \propto \rho \left(1 - e^{-TR/T_1}\right) e^{-TE/T_2}
\]
</div>

In PD-weighted sequences:

- Select **long TR** → eliminate T1 influence
- Select **short TE** → eliminate T2 influence

Final signal approximately:

<div class="math-display">
\[
S_{\text{PD}} \approx \rho
\]
</div>

---

## 4.3 k-space Sampling and Reconstruction

MRI data is first sampled in **k-space (frequency domain)**, rather than directly obtaining images. k-space sampling is directly related to the 2D Fourier transform.

---

**(1) MRI Signal Equation (2D Fourier Encoding)**

The sampling signal of the object magnetization distribution $\rho(x,y)$ under the action of frequency encoding and phase encoding gradients is:

<div class="math-display">
\[
S(k_x, k_y)
=
\iint
\rho(x, y)
\, e^{-j 2\pi (k_x x + k_y y)}
\, dx\, dy
\]
</div>

Where:

- $S(k_x,k_y)$: Acquired k-space data
- $k_x, k_y$: Encoding amounts determined by gradient magnetic fields
- $\rho(x,y)$: Proton density or complex signal of the imaged tissue

This is the standard form of **2D Fourier Transform**.

---

**(2) Inverse Fourier Reconstruction**

Image recovery is achieved by performing inverse Fourier transform on k-space:

<div class="math-display">
\[
\rho(x,y)
=
\iint
S(k_x, k_y)
\, e^{j 2\pi (k_x x + k_y y)}
\, dk_x \, dk_y
\]
</div>

Numerical implementation:

<div class="math-display">
\[
\rho = \mathcal{F}^{-1}\{ S \}
\]
</div>

Clinical MRI equipment uses Fast Fourier Transform (FFT).

# 5. Ultrasound Imaging Key Formulas

Ultrasound imaging is based on the propagation, reflection, and scattering of mechanical waves. Its signal processing chain relies more on wave equations, echo intensity, and envelope detection compared to CT/MRI. This section summarizes the most important mathematical formulas in ultrasound B-mode imaging.

---

## 5.1 Basic Relationships of Sound Wave Propagation

Ultrasound is essentially longitudinal waves (pressure waves). When propagating in tissues, there are basic physical relationships between their velocity, frequency, and wavelength.

---

**(1) Wave Velocity, Frequency, and Wavelength**

The relationship between sound velocity $c$, frequency $f$, and wavelength $\lambda$ is:

<div class="math-display">
\[
c = \lambda f
\]
</div>

Typical tissue sound velocities:

|Tissue|Sound Velocity (m/s)|
| --------------| ----------------|
|Fat|~1450|
|Muscle|~1580|
|Soft tissue average|**1540** (commonly used clinical value)|

---

**(2) Sound Pressure Wave Expression (1D)**

Propagating sound waves can be expressed as:

<div class="math-display">
\[
p(x,t) = p_0 \cos (2\pi f t - kx)
\]
</div>

Where:

- $k = \frac{2\pi}{\lambda}$ is the wave number
- $p_0$ is the pressure amplitude

---

## 5.2 Echo Formation Mechanism

Ultrasound image brightness mainly depends on interface reflection and tissue attenuation.

---

**(1) Tissue Attenuation Model**

After propagating a distance $d$, the amplitude of the sound wave attenuates to:

<div class="math-display">
\[
A(d) = A_0 e^{-\alpha d}
\]
</div>

Or expressed in dB (more common):

<div class="math-display">
\[
A_{\mathrm{dB}}(d) = A_{\mathrm{dB}}(0) - \alpha_{\mathrm{dB}} d
\]
</div>

Where:

- $\alpha$: Linear attenuation coefficient
- $\alpha_{\mathrm{dB}}$: Attenuation constant expressed in dB/cm

Attenuation increases with frequency, so higher frequencies provide better resolution but poorer penetration.

---

**(2) Reflection Coefficient (Acoustic Impedance)**

The reflection intensity at interfaces of different tissues is determined by acoustic impedance:

<div class="math-display">
\[
Z = \rho c
\]
</div>

The greater the difference in acoustic impedance, the stronger the reflection.

The interface reflection coefficient is:

<div class="math-display">
\[
R = \left(
\frac{Z_2 - Z_1}{Z_2 + Z_1}
\right)^2
\]
</div>

Where:

- $Z_1, Z_2$: Acoustic impedance of two tissues
- $R$: Reflection intensity ratio (0~1)

This is the most fundamental source of B-mode imaging brightness.

---

## 5.3 B-mode Imaging Mathematics

B-mode (Brightness mode) is the most common ultrasound imaging method, involving echo envelope extraction and log compression.

---

**(1) Envelope Detection (Hilbert Transform)**

The received signal is typically a radio frequency signal $x(t)$, from which the envelope needs to be extracted:

<div class="math-display">
\[
A(t) = \sqrt{ x^2(t) + \hat{x}^2(t) }
\]
</div>

Where:

- $\hat{x}(t)$: Hilbert transform of $x(t)$
- $A(t)$: Envelope (representing intensity)

The envelope represents the intensity of interface reflections and is the core of B-mode grayscale images.

---

**(2) B-mode Log Compression**

The dynamic range of the original RF envelope is very large (>60 dB) and needs to be compressed to a displayable range:

<div class="math-display">
\[
I_{\text{display}}
=
\log \left( 1 + A(t) \right)
\]
</div>

Or the common dB form:

<div class="math-display">
\[
I_{\mathrm{dB}} = 20 \log_{10}(A)
\]
</div>

Log compression can:

- Compress dynamic range
- Enhance visibility of weakly reflecting tissues
- Improve overall contrast

---

**(3) Scanline to Image**

The final B-mode image is composed of multiple A-scans.

Mathematically, it can be viewed as:

<div class="math-display">
\[
I(x,y) = \text{LogCompress}\left( A(t) \right)
\]
</div>

Where $x$ is determined by the scanning angle and $y$ is the depth (propagation time).

# 6. PET / SPECT Imaging Key Formulas

PET (Positron Emission Tomography) and SPECT (Single Photon Emission Computed Tomography) are based on gamma photons generated by radioactive nuclide decay for imaging. Their mathematical essence includes radioactive decay models, Poisson statistical forward models, and iterative reconstruction (MLEM / OSEM). This section presents the commonly used core formulas for PET/SPECT.

---

## 6.1 Radioactive Decay

The signal source of PET/SPECT is radioactive nuclides, whose tracer activity follows exponential decay.

---

**(1) Radioactive Decay Formula**

<div class="math-display">
\[
N(t) = N_0 e^{-\lambda t}
\]
</div>

Where:

- $N(t)$: Number of nuclides at time $t$
- $N_0$: Initial number of nuclides
- $\lambda$: Decay constant, related to half-life

Half-life formula:

<div class="math-display">
\[
T_{1/2} = \frac{\ln 2}{\lambda}
\]
</div>

---

**(2) Activity**

The decay rate (number of decays per second) is:

<div class="math-display">
\[
A(t) = \lambda N(t)
\]
</div>

Unit is becquerel (Bq).

---

## 6.2 PET Forward Model

The projection data detected by PET/SPECT is essentially a statistical sampling of **radioactive decay → gamma photon emission → detector response**, typically following a **Poisson distribution**.

---

**(1) Poisson Statistical Model**

The counts in detector bin $i$ follow a Poisson distribution:

<div class="math-display">
\[
y_i \sim \text{Poisson}(\lambda_i)
\]
</div>

Where:

- $y_i$: Actual observed count (a pixel in the sinogram)
- $\lambda_i$: Expected count

---

**(2) Projection Data Formation Equation**

The forward projection of PET can be expressed as:

<div class="math-display">
\[
\lambda_i = \sum_{j} a_{ij} x_j + r_i
\]
</div>

Where:

- $x_j$: Radioactivity at pixel $j$ (to be reconstructed)
- $a_{ij}$: System matrix

  - Includes geometric factors, attenuation, scattering, detection efficiency
- $r_i$: Random events, scattering events, background noise

Vector form:

<div class="math-display">
\[
\boldsymbol{\lambda} = A \mathbf{x} + \mathbf{r}
\]
</div>

This is the foundation of PET reconstruction.

---

## 6.3 PET/SPECT Reconstruction (MLEM / OSEM)

PET/SPECT mostly uses statistical **MLEM or OSEM iterative reconstruction**, derived from maximum likelihood estimation (MLE).

---

**(1) MLE (Maximum Likelihood Estimation)**

The likelihood function of observed data $y_i$:

<div class="math-display">
\[
L(\mathbf{x}) = \prod_{i}
\frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!}
\]
</div>

Where $\lambda_i = \sum_j a_{ij} x_j$.

Log-likelihood:

<div class="math-display">
\[
\log L(\mathbf{x})
=
\sum_i \left[
y_i \ln \lambda_i - \lambda_i
\right] + C
\]
</div>

MLE objective:

<div class="math-display">
\[
\hat{\mathbf{x}}
= \arg\max_{\mathbf{x}} \log L(\mathbf{x})
\]
</div>

---

**(2) MLEM (Expectation–Maximization) Update Formula**

The classic MLEM update is:

<div class="math-display">
\[
x_j^{(k+1)}
=
x_j^{(k)}
\frac
{\sum_i a_{ij} \frac{y_i}{\sum_m a_{im}x_m^{(k)}}}
{\sum_i a_{ij}}
\]
</div>

Where:

- First fraction: Data consistency term
- Denominator: Normalization factor
- Form similar to "forward projection → normalization → backprojection"

MLEM has good convergence but is slow.

---

**(3) OSEM (Ordered Subsets Expectation Maximization)**

OSEM divides the sinogram into multiple subsets, using only partial projections for each update to improve speed.

OSEM update form:

<div class="math-display">
\[
x_j^{(k+1)}
=
x_j^{(k)}
\frac
{\sum_{i \in S_k} a_{ij} \frac{y_i}{\sum_m a_{im}x_m^{(k)}}}
{\sum_{i \in S_k} a_{ij}}
\]
</div>

Where $S_k$ is the k-th subset.

- Speeds up reconstruction (approximately S times faster than MLEM)
- Commonly used in clinical PET/SPECT reconstruction

# 7. X-ray / DR Imaging Key Formulas

The core of X-ray / DR (Digital Radiography) imaging comes from the attenuation of photons passing through tissues and detector response. Its physical model is usually composed of **Lambert-Beer law + line integral projection model + detector gain correction**. This section summarizes the most important formulas in DR imaging.

---

## 7.1 Projection Model (Line Integral)

In DR (projection radiography), X-rays pass through the object from a single direction, and the attenuation along the penetration path constitutes the pixel values of the 2D image.

---

**(1) Photon Attenuation Model (Lambert-Beer)**

For incident intensity $I_0$, the transmitted intensity $I$ is:

<div class="math-display">
\[
I = I_0 \exp
\left(
-\int_L \mu(s) \, ds
\right)
\]
</div>

Where:

- $\mu(s)$: Attenuation coefficient at position $s$ along the path
- $L$: Path the light ray passes through

---

**(2) Projection (Line Integral) Model**

Define the projection:

<div class="math-display">
\[
p = \int_L \mu(s)\, ds
\]
</div>

Then the ideal brightness model of DR image can be expressed as:

<div class="math-display">
\[
I = I_0 e^{-p}
\]
</div>

After log-transform, we get a linear expression:

<div class="math-display">
\[
p = -\ln \left(\frac{I}{I_0}\right)
\]
</div>

Meaning:

- DR is a single-angle special case of Radon transform (CT is multi-angle)
- p reflects the overall attenuation of X-rays, determining imaging brightness

---

**(3) Brightness and Attenuation Relationship (Direct Representation)**

Brightness (transmittance) is usually negatively correlated with attenuation:

<div class="math-display">
\[
\text{Brightness}(x,y) \propto e^{-\mu(x,y) d}
\]
</div>

Where $d$ is the tissue thickness.

---

## 7.2 Image Normalization and Gain Correction

Real detectors have non-uniformity, dark current, and gain deviation, so DR images must undergo flat-field correction to restore correct attenuation signatures.

---

**(1) Detector Response Model**

The original DR image can be expressed as:

<div class="math-display">
\[
I_{\text{raw}}
=
G \cdot I_{\text{signal}} + D
\]
</div>

Where:

- $G$: Pixel gain (gain), varies with pixels
- $D$: Dark current (dark field)
- $I_{\text{signal}}$: Ideal signal

---

**(2) Flat-field Correction**

The formula for flat-field correction is:

<div class="math-display">
\[
I_{\text{corr}}
=
\frac{ I_{\text{raw}} - I_{\text{dark}} }
     { I_{\text{flat}} - I_{\text{dark}} }
\]
</div>

Where:

- $I_{\text{raw}}$: Original acquired image
- $I_{\text{flat}}$: Flat field image under uniform illumination
- $I_{\text{dark}}$: Dark field image without illumination

Meaning:

- Correct gain non-uniformity (pixel gain variation)
- Correct dark current bias
- Restore correct ray transmission information

---

**(3) Linear Normalization (for Visualization)**

Corrected images are often normalized:

<div class="math-display">
\[
I_{\text{norm}} =
\frac{I_{\text{corr}} - \min(I_{\text{corr}})}
     {\max(I_{\text{corr}})-\min(I_{\text{corr}})}
\]
</div>

Used for:

- Display optimization
- Subsequent image processing / machine learning algorithms

# 8. Common Formulas for Image Processing and Enhancement

In medical image processing, preprocessing, enhancement, and feature computation are fundamental steps commonly relied upon by CT, MRI, ultrasound, PET, X-ray, and other modalities. This section summarizes common standardization methods, filters, edge operators, as well as affine transformations and interpolation formulas in resampling.

---

## 8.1 Histogram and Normalization

Standardization is a key step to improve image comparability and model robustness.

---

**(1) Min-Max Normalization**

Linearly map values to $[0,1]$:

<div class="math-display">
\[
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]
</div>

Suitable for grayscale normalization, window width mapping (such as CT images).

---

**(2) Z-score Standardization**

Commonly used for input standardization of deep learning models:

<div class="math-display">
\[
x' = \frac{x - \mu}{\sigma}
\]
</div>

Where:

- $\mu$: Mean
- $\sigma$: Standard deviation

Makes data satisfy zero mean and unit variance.

---

**(3) Histogram Equalization**

Enhances contrast by making the grayscale distribution more uniform.

CDF (Cumulative Distribution Function) is:

<div class="math-display">
\[
\text{CDF}(x)=\sum_{i=0}^{x} \frac{h(i)}{N}
\]
</div>

Equalized pixels:

<div class="math-display">
\[
x' = (L-1) \cdot \text{CDF}(x)
\]
</div>

Where:

- $h(i)$: Frequency of grayscale i
- $N$: Total number of pixels
- $L$: Number of grayscale levels (usually 256)

Suitable for DR, ultrasound, and some MRI scenarios.

---

## 8.2 Filtering and Feature Computation

Image filtering is used for smoothing, denoising, and edge enhancement, and is a fundamental operation in image processing.

---

**(1) Gaussian Filter**

1D Gaussian kernel:

<div class="math-display">
\[
G(x) =
\frac{1}{\sqrt{2\pi\sigma^2}}
e^{ -\frac{x^2}{2\sigma^2} }
\]
</div>

2D Gaussian kernel:

<div class="math-display">
\[
G(x,y) =
\frac{1}{2\pi\sigma^2}
e^{ -\frac{x^2+y^2}{2\sigma^2} }
\]
</div>

Functions:

- Smooth images
- Reduce high-frequency noise
- Commonly used for denoising preprocessing (such as CT, X-ray, Ultrasound)

---

**(2) Sobel Edge Operator**

Used to detect horizontal or vertical edges.

Sobel-x:

<div class="math-display">
\[
G_x =
\begin{bmatrix}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1
\end{bmatrix}
\]
</div>

Sobel-y:

<div class="math-display">
\[
G_y =
\begin{bmatrix}
-1 & -2 & -1\\
0 & 0 & 0\\
1 & 2 & 1
\end{bmatrix}
\]
</div>

Gradient intensity:

<div class="math-display">
\[
|\nabla I| = \sqrt{ (G_x * I)^2 + (G_y * I)^2 }
\]
</div>

Suitable for edge detection and morphological analysis.

---

**(3) Laplacian Edge Enhancement**

Second-order derivative operator:

<div class="math-display">
\[
\nabla^2 I =
\frac{\partial^2 I}{\partial x^2}
+
\frac{\partial^2 I}{\partial y^2}
\]
</div>

Typical discrete template:

<div class="math-display">
\[
\begin{bmatrix}
0 & -1 & 0\\
-1 & 4 & -1\\
0 & -1 & 0
\end{bmatrix}
\]
</div>

Can be used for sharpening or edge enhancement.

---

## 8.3 Resampling and Alignment

Medical image alignment is often based on affine transformations and spatial interpolation.

---

**(1) Affine Transformation Matrix**

3D affine transformation (commonly used for CT/MRI registration):

<div class="math-display">
\[
\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{b}
\]
</div>

Homogeneous form:

<div class="math-display">
\[
\begin{bmatrix}
\mathbf{x}' \\ 1
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{A} & \mathbf{b}\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\mathbf{x} \\ 1
\end{bmatrix}
\]
</div>

Where:

- $\mathbf{A}$: Contains rotation, scaling, shearing
- $\mathbf{b}$: Translation vector

Used for:

- Multimodal registration (MRI ↔ CT)
- Image alignment
- Data standardization (e.g., unified voxel spacing)

---

**(2) Trilinear Interpolation**

The most commonly used method for resampling 3D volumetric data.

General formula for trilinear interpolation:

<div class="math-display">
\[
f(x,y,z)
=
\sum_{i=0}^1\sum_{j=0}^1\sum_{k=0}^1
f(i,j,k)
(1-|x-i|)
(1-|y-j|)
(1-|z-k|)
\]
</div>

Meaning:

- Each point is weighted by the surrounding 8 voxels
- Weights are linearly determined by distance

Suitable for:

- Volumetric data resampling
- CT/MRI spacing normalization
- Resample steps in spatial transformation

# 9. Key Formulas in Deep Learning (Commonly Used in Medical Imaging)

Deep learning has been widely applied to segmentation, classification, detection, and reconstruction tasks in medical imaging. This section summarizes the most commonly used convolution formulas, loss functions, and evaluation metrics in medical imaging tasks, providing a mathematical foundation for understanding subsequent algorithms.

---

## 9.1 Convolutional Layers (2D / 3D)

Convolution is the most important operation of CNN in medical imaging (e.g., 2D MRI slices, 3D CT volumetric data, ultrasound sequences, etc.).

---

**(1) Convolution Operation Formula**

● 2D Convolution (Image)

<div class="math-display">
\[
y(i,j) = \sum_m \sum_n x(i-m, j-n)\, k(m,n)
\]
</div>

● 3D Convolution (Volumetric Data)

<div class="math-display">
\[
y(i,j,k)
=
\sum_{u} \sum_{v} \sum_{w}
x(i-u, j-v, k-w)\, k(u,v,w)
\]
</div>

3D convolution is widely used in CT/MRI segmentation and 3D detection tasks.

---

**(2) Convolution Output Size Calculation Formula**

● 2D Output Size

<div class="math-display">
\[
H_{\text{out}} =
\frac{H_{\text{in}} - K + 2P}{S} + 1
\]
</div>

<div class="math-display">
\[
W_{\text{out}} =
\frac{W_{\text{in}} - K + 2P}{S} + 1
\]
</div>

● 3D Output Size

<div class="math-display">
\[
D_{\text{out}} =
\frac{D_{\text{in}} - K + 2P}{S} + 1
\]
</div>

Parameter description:

- $K$: Kernel size
- $S$: Stride
- $P$: Padding
- Input/output dimensions are used for CNN structure design (UNet, VNet, etc.)

---

## 9.2 Loss Functions

Medical imaging tasks often involve class imbalance, so Dice Loss and Focal Loss are particularly commonly used in segmentation and detection tasks.

---

**(1) Dice Loss (Commonly Used for Segmentation)**

Dice coefficient:

<div class="math-display">
\[
\text{Dice} =
\frac{2|A \cap B|}{|A|+|B|}
\]
</div>

Dice Loss:

<div class="math-display">
\[
\mathcal{L}_{Dice} = 1 - \text{Dice}
\]
</div>

Used for segmentation tasks (especially organs, lesions, small targets).

---

**(2) Cross-Entropy Loss (Foundation for Classification/Segmentation)**

● Binary Cross-Entropy

<div class="math-display">
\[
\mathcal{L}_{CE}
=
- \left[
y \log(\hat{y}) + (1-y)\log(1-\hat{y})
\right]
\]
</div>

● Multi-class Cross-Entropy

<div class="math-display">
\[
\mathcal{L}_{CE}
=
-\sum_{c=1}^{C}
y_c \log(\hat{y}_c)
\]
</div>

---

**(3) Focal Loss (Solving Class Imbalance)**

Focal Loss suppresses easy samples and highlights difficult samples:

<div class="math-display">
\[
\mathcal{L}_{Focal}
=
-(1-\hat{y})^\gamma \, y\log(\hat{y})
\]
</div>

Where:

- $\gamma$: Adjusts the weight of difficult samples (typical values 1–3)
- Commonly used in class-imbalanced tasks such as lung nodule detection and tumor detection

---

## 9.3 Evaluation Metrics (Segmentation & Classification Metrics)

Evaluation metrics reflect the performance of models in medical imaging tasks, especially critical in lesion segmentation, organ segmentation, and pathology classification.

---

**(1) Dice Coefficient**

<div class="math-display">
\[
\text{Dice} =
\frac{2|A \cap B|}{|A|+|B|}
\]
</div>

- 0 (poor) → 1 (perfect)
- Commonly used for CT/MRI organ/lesion segmentation evaluation

---

**(2) IoU (Intersection over Union)**

<div class="math-display">
\[
\text{IoU}=
\frac{|A \cap B|}{|A \cup B|}
\]
</div>

Relationship with Dice:

<div class="math-display">
\[
\text{Dice} = \frac{2\text{IoU}}{1+\text{IoU}}
\]
</div>

---

**(3) Sensitivity**

<div class="math-display">
\[
\text{Sensitivity}=
\frac{TP}{TP+FN}
\]
</div>

Measures the ability to "detect lesions".

---

**(4) Specificity**

<div class="math-display">
\[
\text{Specificity}=
\frac{TN}{TN+FP}
\]
</div>

Measures the ability to "avoid false positives".

