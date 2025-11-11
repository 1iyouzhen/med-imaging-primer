#!/usr/bin/env python3
"""
Simplified Medical Image Segmentation Augmentation Demo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import gaussian
import os
from pathlib import Path

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

def create_sample_medical_image():
    """
    Create sample medical image (simulated CT lung field)
    """
    print("Creating sample medical image...")

    # Create 512x512 image
    image = np.zeros((512, 512), dtype=np.float32)
    mask = np.zeros((512, 512), dtype=np.float32)

    # Create lung field areas (elliptical shapes)
    y, x = np.ogrid[:512, :512]

    # Left lung
    center_left = (200, 256)
    axes_left = (120, 160)
    angle_left = 30

    # Right lung
    center_right = (312, 256)
    axes_right = (120, 160)
    angle_right = -30

    # Create lung masks
    # Simplified circular lung fields
    left_lung = ((x - center_left[0])**2 + (y - center_left[1])**2) <= (120**2)
    right_lung = ((x - center_right[0])**2 + (y - center_right[1])**2) <= (120**2)

    lung_mask = (left_lung | right_lung).astype(np.float32)

    # Add lung texture
    noise = np.random.randn(512, 512) * 0.1
    texture = gaussian(noise, sigma=2)

    # Lung density (HU value simulation)
    lung_density = -800 + texture * 100  # Lung HU value ~-800
    body_density = 0  # Soft tissue HU value ~0

    # Combine image
    image = np.where(lung_mask > 0.5, lung_density, body_density)

    # Add small nodule
    nodule_center = (250, 200)
    nodule_radius = 15
    nodule_mask = ((x - nodule_center[0])**2 + (y - nodule_center[1])**2) <= nodule_radius**2
    nodule_density = -300 + np.random.randn() * 50  # Nodule density
    image = np.where(nodule_mask, nodule_density, image)

    # Normalize to 0-255 for display
    image_display = ((image + 1000) / 1000 * 255).clip(0, 255).astype(np.uint8)
    mask_display = (lung_mask * 255).astype(np.uint8)

    print(f"  Image size: {image.shape}")
    print(f"  Lung ratio: {lung_mask.mean():.2%}")
    print(f"  Density range: [{image.min():.1f}, {image.max():.1f}] HU")

    return image, image_display, lung_mask, mask_display, nodule_mask

def elastic_deformation(image, mask, alpha=1000, sigma=8):
    """
    Elastic deformation: Simulate physiological changes like breathing
    """
    print(f"Applying elastic deformation (alpha={alpha}, sigma={sigma})")

    shape = image.shape

    # Generate random displacement field
    dx = gaussian(np.random.randn(*shape), sigma, mode='reflect') * alpha
    dy = gaussian(np.random.randn(*shape), sigma, mode='reflect') * alpha

    # Create grid coordinates
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0], dtype=np.float32))

    # Apply displacement
    indices = np.array([y + dy, x + dx])

    # Warp image and mask using scipy
    warped_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
    warped_mask = ndimage.map_coordinates(mask, indices, order=0, mode='reflect')

    return warped_image, warped_mask

def intensity_transform(image, mask, contrast_factor=1.2, brightness_shift=50):
    """
    Intensity transformation: Simulate different scanning parameters
    """
    print(f"Applying intensity transformation (contrast={contrast_factor}, brightness={brightness_shift})")

    # Apply contrast and brightness adjustment
    transformed = image * contrast_factor + brightness_shift

    # Keep HU value range reasonable
    transformed = np.clip(transformed, -1000, 1000)

    return transformed, mask

def add_noise(image, mask, noise_level=20):
    """
    Add noise: Simulate real clinical environment image noise
    """
    print(f"Adding Gaussian noise (level={noise_level})")

    # Gaussian noise (simulate electronic noise)
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise

    return noisy_image, mask

def add_metal_artifacts(image, mask, severity=0.3):
    """
    Add metal artifacts: Simulate metal artifact effects
    """
    print(f"Adding metal artifacts (severity={severity})")

    artifacted_image = image.copy()

    # Metal artifacts (linear streaks)
    n_lines = int(5 * severity)
    for _ in range(n_lines):
        start_x = np.random.randint(100, 400)
        start_y = 0
        end_x = start_x + np.random.randint(-20, 20)
        end_y = image.shape[0]

        # Create line mask
        y_coords = np.arange(start_y, end_y)
        x_coords = np.linspace(start_x, end_x, len(y_coords)).astype(int)
        x_coords = np.clip(x_coords, 0, image.shape[1]-1)

        # Add metal artifacts (high density streaks)
        for y, x in zip(y_coords, x_coords):
            if 0 <= y < image.shape[0]:
                # Streak width
                for dx in range(-2, 3):
                    if 0 <= x+dx < image.shape[1]:
                        artifacted_image[y, x+dx] = 2000  # Metal density HU value

    return artifacted_image, mask

def visualize_augmentation_results(original_img, original_mask, augmentations):
    """
    Visualize augmentation results
    """
    print("Generating augmentation visualization...")

    # Calculate display range
    all_images = [original_img] + [aug[0] for aug in augmentations]
    vmin = min(img.min() for img in all_images)
    vmax = max(img.max() for img in all_images)

    # Create 8-panel layout
    fig = plt.figure(figsize=(20, 12))

    # Original image
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(original_img, cmap='gray', vmin=vmin, vmax=vmax)
    ax1.set_title('Original Image\n(Simulated CT Lung Field)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='HU Value')

    # Original mask
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(original_mask, cmap='Reds')
    ax2.set_title('Lung Mask\n(Ground Truth)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Overlay display
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(original_img, cmap='gray', vmin=vmin, vmax=vmax)
    ax3.imshow(original_mask, cmap='Reds', alpha=0.3)
    ax3.set_title('Image + Mask Overlay\n(Nodule Location)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Statistics
    ax4 = plt.subplot(2, 4, 4)
    ax4.axis('off')
    stats_text = f"""Original Image Statistics:

Size: {original_img.shape}
Min: {original_img.min():.1f} HU
Max: {original_img.max():.1f} HU
Mean: {original_img.mean():.1f} HU
Std: {original_img.std():.1f} HU

Lung Ratio: {original_mask.mean():.2%}
Nodule Size: {np.sum(original_mask>0):.0f} pixels"""

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_title('Image Information', fontsize=12, fontweight='bold')

    # Augmentation results
    augmentation_names = [
        "Elastic Deformation\n(Simulate Breathing Motion)",
        "Intensity Transform\n(Contrast Adjustment)",
        "Noise Addition\n(Gaussian Noise)",
        "Metal Artifacts\n(Clinical Artifacts)"
    ]

    for i, (augmented_img, augmented_mask, aug_name) in enumerate(augmentations):
        ax = plt.subplot(2, 4, i+5)

        # Display augmented image
        im = ax.imshow(augmented_img, cmap='gray', vmin=vmin, vmax=vmax)
        # Overlay mask
        ax.imshow(augmented_mask, cmap='Reds', alpha=0.3)

        ax.set_title(augmentation_names[i], fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save image
    output_path = output_dir / "medical_segmentation_augmentation_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    plt.show()
    plt.pause(2)
    plt.close()

    return output_path

def main():
    """
    Main function
    """
    print("=" * 80)
    print("Medical Image Segmentation Augmentation Demo")
    print("=" * 80)

    # Create sample medical image
    print("\nCreating sample medical image...")
    original_img, original_img_display, original_mask, original_mask_display, nodule_mask = create_sample_medical_image()

    # Apply different augmentation techniques
    print("\nApplying augmentation techniques...")
    augmentations = []

    # 1. Elastic deformation
    elastic_img, elastic_mask = elastic_deformation(
        original_img, original_mask, alpha=800, sigma=6
    )
    augmentations.append((elastic_img, elastic_mask, "Elastic deformation"))

    # 2. Intensity transformation
    intensity_img, intensity_mask = intensity_transform(
        original_img, original_mask, contrast_factor=1.3, brightness_shift=50
    )
    augmentations.append((intensity_img, intensity_mask, "Intensity transform"))

    # 3. Noise addition
    noise_img, noise_mask = add_noise(
        original_img, original_mask, noise_level=15
    )
    augmentations.append((noise_img, noise_mask, "Noise addition"))

    # 4. Metal artifacts
    artifact_img, artifact_mask = add_metal_artifacts(
        original_img, original_mask, severity=0.4
    )
    augmentations.append((artifact_img, artifact_mask, "Metal artifacts"))

    # Visualize results
    print("\nGenerating visualization results...")
    viz_path = visualize_augmentation_results(
        original_img_display, original_mask_display, augmentations
    )

    print("\n" + "=" * 80)
    print("Medical Segmentation Augmentation Demo Completed!")
    print("=" * 80)

    return str(viz_path)

if __name__ == "__main__":
    result_path = main()
    print(f"\nDemo completed. Visualization saved to: {result_path}")