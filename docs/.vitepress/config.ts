import { defineConfig } from 'vitepress'
import { figure } from '@mdit/plugin-figure'
import { MermaidPlugin } from './theme/plugins/mermaid'

export default defineConfig({
  title: '医学影像处理开源教程',
  description: '从物理成像原理到深度学习的系统性入门指南',

  // GitHub Pages 部署路径（仓库名）
  base: '/med-imaging-primer/',

  // 路径重写：将 zh/ 目录映射到根路径
  rewrites: {
    'zh/:rest*': ':rest*'
  },

  // 国际化配置
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: '医学影像处理开源教程',
      description: '从物理成像原理到深度学习的系统性入门指南',
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          { text: '教程', link: '/guide/' },
          {
            text: 'GitHub',
            link: 'https://github.com/datawhalechina/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/guide/': [
            {
              text: '导览',
              link: '/guide/'
            },
            {
              text: '第1章 医学影像基础',
              collapsed: true,
              items: [
                {
                  text: '1.1 常见成像模态原理与特点',
                  collapsed: false,
                  items: [
                    { text: '1.1.1 CT（计算机断层扫描）', link: '/guide/ch01/01-modalities/01-ct' },
                    { text: '1.1.2 MRI（磁共振成像）', link: '/guide/ch01/01-modalities/02-mri' },
                    { text: '1.1.3 X射线成像', link: '/guide/ch01/01-modalities/03-xray' },
                    { text: '1.1.4 PET与超声', link: '/guide/ch01/01-modalities/04-pet-us' }
                  ]
                },
                { text: '1.2 数据格式标准', link: '/guide/ch01/02-data-formats' },
                { text: '1.3 常用开源工具', link: '/guide/ch01/03-tools' },
                { text: '1.4 图像质量与典型伪影', link: '/guide/ch01/04-artifacts' }
              ]
            },
            {
              text: '第2章 重建前处理：模态特异性校正流程',
              collapsed: true,
              items: [
                { text: '2.1 CT：从探测器信号到校正投影', link: '/guide/ch02/01-ct-preprocessing' },
                { text: '2.2 MRI：k空间数据预处理', link: '/guide/ch02/02-mri-preprocessing' },
                { text: '2.3 X-ray：直接成像的校正', link: '/guide/ch02/03-xray-preprocessing' }
              ]
            },
            {
              text: '第3章 图像重建算法（按模态组织）',
              collapsed: true,
              items: [
                { text: '3.1 CT重建', link: '/guide/ch03/01-ct-reconstruction' },
                { text: '3.2 MRI重建', link: '/guide/ch03/02-mri-reconstruction' },
                { text: '3.3 X-ray成像', link: '/guide/ch03/03-xray-imaging' }
              ]
            },
            {
              text: '第4章 重建实践与验证（多模态示例）',
              collapsed: true,
              items: [
                { text: '4.1 CT完整流程', link: '/guide/ch04/01-ct-workflow' },
                { text: '4.2 MRI重建示例', link: '/guide/ch04/02-mri-example' },
                { text: '4.3 X-ray校正示例', link: '/guide/ch04/03-xray-example' },
                { text: '4.4 重建质量评估', link: '/guide/ch04/04-quality-assessment' },
                { text: '4.5 常见问题排查指南', link: '/guide/ch04/05-troubleshooting' }
              ]
            },
            {
              text: '第5章 医学图像后处理（通用+模态适配）',
              collapsed: true,
              items: [
                { text: '5.1 预处理（强调模态差异）', link: '/guide/ch05/01-preprocessing' },
                { text: '5.2 图像分割：U-Net 及其变体', link: '/guide/ch05/02-segmentation' },
                { text: '5.3 分类与检测', link: '/guide/ch05/03-classification' },
                { text: '5.4 图像增强与恢复', link: '/guide/ch05/04-enhancement' }
              ]
            },
            {
              text: '附录',
              collapsed: true,
              items: [
                { text: '附录A：关键公式', link: '/guide/appendix/A-formula' },
                { text: '附录B：工具安装', link: '/guide/appendix/B-tool-Installation' },
                {
                  text: '附录C：公开数据集列表',
                  collapsed: true,
                  items: [
                    { text: 'C.1 CT数据集', link: '/guide/appendix/C-dataset/C-1-CT' },
                    { text: 'C.2 MRI数据集', link: '/guide/appendix/C-dataset/C-2-MRI' },
                    { text: 'C.3 X射线数据集', link: '/guide/appendix/C-dataset/C-3-X-ray' },
                    { text: 'C.4 多模态数据集', link: '/guide/appendix/C-dataset/C-4-Multimodal' }
                  ]
                },
                { text: '附录D：术语表', link: '/guide/appendix/D-glossary' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/med-imaging-primer' }
        ],
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2025-present Your Name'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      title: 'Medical Imaging Primer',
      description: 'An Open Primer on Medical Imaging: From Physics to Deep Learning',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Tutorial', link: '/en/guide/' },
          {
            text: 'GitHub',
            link: 'https://github.com/datawhalechina/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/en/guide/': [
            {
              text: 'Introduction',
              link: '/en/guide/'
            },
            {
              text: 'Chapter 1: Medical Imaging Basics',
              collapsed: true,
              items: [
                {
                  text: '1.1 Common Imaging Modality Principles',
                  collapsed: false,
                  items: [
                    { text: '1.1.1 CT (Computed Tomography)', link: '/en/guide/ch01/01-modalities/01-ct' },
                    { text: '1.1.2 MRI (Magnetic Resonance Imaging)', link: '/en/guide/ch01/01-modalities/02-mri' },
                    { text: '1.1.3 X-ray Imaging', link: '/en/guide/ch01/01-modalities/03-xray' },
                    { text: '1.1.4 PET & Ultrasound', link: '/en/guide/ch01/01-modalities/04-pet-us' }
                  ]
                },
                { text: '1.2 Data Format Standards', link: '/en/guide/ch01/02-data-formats' },
                { text: '1.3 Common Open-Source Tools', link: '/en/guide/ch01/03-tools' },
                { text: '1.4 Image Quality and Typical Artifacts', link: '/en/guide/ch01/04-artifacts' }
              ]
            },
            {
              text: 'Chapter 2: Pre-Reconstruction Processing',
              collapsed: true,
              items: [
                { text: '2.1 CT: From Detector Signal to Corrected Projection', link: '/en/guide/ch02/01-ct-preprocessing' },
                { text: '2.2 MRI: k-Space Data Preprocessing', link: '/en/guide/ch02/02-mri-preprocessing' },
                { text: '2.3 X-ray: Direct Imaging Correction', link: '/en/guide/ch02/03-xray-preprocessing' }
              ]
            },
            {
              text: 'Chapter 3: Image Reconstruction Algorithms',
              collapsed: true,
              items: [
                { text: '3.1 CT Reconstruction', link: '/en/guide/ch03/01-ct-reconstruction' },
                { text: '3.2 MRI Reconstruction', link: '/en/guide/ch03/02-mri-reconstruction' },
                { text: '3.3 X-ray Imaging', link: '/en/guide/ch03/03-xray-imaging' }
              ]
            },
            {
              text: 'Chapter 4: Reconstruction Practice and Validation',
              collapsed: true,
              items: [
                { text: '4.1 CT Complete Workflow', link: '/en/guide/ch04/01-ct-workflow' },
                { text: '4.2 MRI Reconstruction Example', link: '/en/guide/ch04/02-mri-example' },
                { text: '4.3 X-ray Correction Example', link: '/en/guide/ch04/03-xray-example' },
                { text: '4.4 Reconstruction Quality Assessment', link: '/en/guide/ch04/04-quality-assessment' },
                { text: '4.5 Common Issues Troubleshooting Guide', link: '/en/guide/ch04/05-troubleshooting' }
              ]
            },
            {
              text: 'Chapter 5: Medical Image Post-Processing',
              collapsed: true,
              items: [
                { text: '5.1 Preprocessing (Modality-Specific)', link: '/en/guide/ch05/01-preprocessing' },
                { text: '5.2 Image Segmentation: U-Net and its Variants', link: '/en/guide/ch05/02-segmentation' },
                { text: '5.3 Classification and Detection', link: '/en/guide/ch05/03-classification' },
                { text: '5.4 Image Enhancement and Restoration', link: '/en/guide/ch05/04-enhancement' }
              ]
            },
            {
              text: 'Appendix',
              collapsed: true,
              items: [
                { text: 'Appendix A: Key Formulas', link: '/en/guide/appendix/A-formula' },
                { text: 'Appendix B: Tool Installation', link: '/en/guide/appendix/B-tool-Installation' },
                {
                  text: 'Appendix C: Public Datasets',
                  collapsed: true,
                  items: [
                    { text: 'C.1 CT Datasets', link: '/en/guide/appendix/C-dataset/C-1-CT' },
                    { text: 'C.2 MRI Datasets', link: '/en/guide/appendix/C-dataset/C-2-MRI' },
                    { text: 'C.3 X-ray Datasets', link: '/en/guide/appendix/C-dataset/C-3-X-ray' },
                    { text: 'C.4 Multimodal Datasets', link: '/en/guide/appendix/C-dataset/C-4-Multimodal' }
                  ]
                },
                { text: 'Appendix D: Glossary', link: '/en/guide/appendix/D-glossary' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/med-imaging-primer' }
        ],
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2025-present Your Name'
        }
      }
    }
  },

  // Markdown 扩展
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true,
    config: (md) => {
      md.use(figure)
      md.use(MermaidPlugin)
    },
    math: true
  },
  themeConfig: {
    search: {
      provider: 'local'
    }
  }
})

