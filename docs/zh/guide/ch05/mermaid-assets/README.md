# Chapter 5 Mermaid 资源文件夹

本文件夹包含第五章所有 Mermaid 图表的源文件和渲染后的高清图片。

## 📁 文件夹结构

```
mermaid-assets/
├── source-files/          # Mermaid 源文件 (.mmd)
│   ├── 01-preprocessing-hierarchy-zh.mmd
│   ├── 02-preprocessing-strategy-zh.mmd
│   ├── 03-unet-architecture-zh.mmd
│   ├── 04-unet-plus-plus-zh.mmd
│   └── 05-model-selection-zh.mmd
├── rendered-images/       # 渲染后的高清图片
│   ├── 01-preprocessing-hierarchy-zh.png
│   ├── 02-preprocessing-strategy-zh.png
│   ├── 03-unet-architecture-zh.png
│   ├── 04-unet-plus-plus-zh.png
│   └── 05-model-selection-zh.png
└── README.md             # 本说明文件
```

## 🎨 图表列表

| 序号 | 图表名称 | 描述 | 尺寸 |
|------|----------|------|------|
| 01 | 医学影像预处理层次结构 | 展示预处理的三个层次 | 1200×800 |
| 02 | 任务驱动的预处理策略 | 根据模态选择预处理流程 | 1200×900 |
| 03 | U-Net架构深度解析 | U-Net编码器-解码器结构 | 1400×1000 |
| 04 | U-Net++密集跳跃连接 | U-Net++的密集连接结构 | 1000×700 |
| 05 | 医学图像分析模型选择指南 | 根据数据类型选择模型 | 1400×1000 |

## 🔧 技术规格

- **渲染工具**: Mermaid CLI (mmdc) v11.12.0
- **图片格式**: PNG
- **分辨率**: 高DPI (2倍缩放)
- **字体支持**: 系统字体，兼容中英文
- **颜色方案**: 优化的对比度配色
- **样式特性**:
  - 双语标注 (中文/英文)
  - 颜色编码的模块区分
  - 粗体强调重要概念
  - 渐变色彩层次

## 📝 使用方法

### 在 Markdown 中引用

```markdown
![图表名称](/images/ch05/01-preprocessing-hierarchy-zh.png)
*图：图表描述*[📄 [Mermaid源文件](/images/ch05/01-preprocessing-hierarchy-zh.mmd)]
```

### 渲染新图片

如需重新渲染图片，使用以下命令：

```bash
cd source-files
mmdc -i "01-preprocessing-hierarchy-zh.mmd" -o "../rendered-images/01-preprocessing-hierarchy-zh.png" -w 1200 -H 800 -s 2
```

### 编辑源文件

1. 直接编辑 `.mmd` 文件
2. 重新渲染对应的图片
3. 更新 Markdown 文件中的引用

## 🎯 设计原则

- **双语支持**: 所有图表同时支持中英文显示
- **可读性**: 高对比度颜色，清晰的层次结构
- **一致性**: 统一的配色方案和字体样式
- **专业性**: 符合医学影像学术标准
- **可维护性**: 模块化设计，易于修改和扩展

## 🔄 版本控制

- 源文件 (`.mmd`) 纳入版本控制
- 渲染图片定期更新，保持与源文件同步
- 重大修改时更新版本号和修改日期

## 📞 联系方式

如有问题或建议，请通过项目仓库提交 Issue。