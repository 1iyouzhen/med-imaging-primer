# Chapter 5 Mermaid Assets Folder

This folder contains all Mermaid diagram source files and rendered high-DPI images for Chapter 5.

## 📁 Folder Structure

```
mermaid-assets/
├── source-files/          # Mermaid source files (.mmd)
│   ├── 01-preprocessing-hierarchy-en.mmd
│   ├── 02-preprocessing-strategy-en.mmd
│   ├── 03-unet-architecture-en.mmd
│   ├── 04-unet-plus-plus-en.mmd
│   └── 05-classification-detection-en.mmd
├── rendered-images/       # Rendered high-DPI images
│   ├── 01-preprocessing-hierarchy-en.png
│   ├── 02-preprocessing-strategy-en.png
│   ├── 03-unet-architecture-en.png
│   ├── 04-unet-plus-plus-en.png
│   └── 05-classification-detection-en.png
└── README.md             # This documentation
```

## 🎨 Diagram List

| # | Diagram Name | Description | Dimensions |
|---|--------------|-------------|-------------|
| 01 | Medical Image Preprocessing Hierarchy | Three levels of medical image preprocessing | 1200×800 |
| 02 | Task-driven Preprocessing Strategy | Modality-based preprocessing workflow | 1200×900 |
| 03 | U-Net Architecture Deep Dive | U-Net encoder-decoder structure | 1400×1000 |
| 04 | U-Net++ Dense Skip Connections | Dense connection structure of U-Net++ | 1000×700 |
| 05 | Classification and Detection Models | Model selection based on data types | 1200×900 |

## 🔧 Technical Specifications

- **Rendering Tool**: Mermaid CLI (mmdc) v11.12.0
- **Image Format**: PNG
- **Resolution**: High-DPI (2x scaling)
- **Font Support**: System fonts, compatible with English text
- **Color Scheme**: Optimized contrast color palette
- **Style Features**:
  - Bilingual labels (English/Chinese)
  - Color-coded module differentiation
  - Bold emphasis on important concepts
  - Gradient color layering

## 📝 Usage Guide

### Reference in Markdown

```markdown
![Diagram Name](/images/ch05/01-preprocessing-hierarchy-en.png)
*Figure: Diagram description*[📄 [Mermaid Source](/images/ch05/01-preprocessing-hierarchy-en.mmd)]
```

### Render New Images

To re-render images, use the following command:

```bash
cd source-files
mmdc -i "01-preprocessing-hierarchy-en.mmd" -o "../rendered-images/01-preprocessing-hierarchy-en.png" -w 1200 -H 800 -s 2
```

### Edit Source Files

1. Edit `.mmd` files directly
2. Re-render corresponding images
3. Update references in Markdown files

## 🎯 Design Principles

- **Bilingual Support**: All diagrams support both English and Chinese
- **Readability**: High contrast colors, clear hierarchical structure
- **Consistency**: Unified color scheme and font styles
- **Professionalism**: Meets medical imaging academic standards
- **Maintainability**: Modular design, easy to modify and extend

## 🔄 Version Control

- Source files (`.mmd`) are under version control
- Rendered images are updated periodically to sync with source files
- Update version numbers and modification dates for major changes

## 📞 Contact

For questions or suggestions, please submit an Issue through the project repository.