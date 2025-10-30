import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Medical Imaging Primer',
  description: 'An Open Primer on Medical Imaging: From Physics to Deep Learning',

  // GitHub Pages 部署路径（仓库名）
  base: '/med-imaging-primer/',

  // 路径重写：将 en/ 目录映射到根路径
  rewrites: {
    'en/:rest*': ':rest*'
  },

  // 国际化配置
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      title: 'Medical Imaging Primer',
      description: 'An Open Primer on Medical Imaging: From Physics to Deep Learning',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Tutorial', link: '/guide/ch01/01-modalities/01-ct' },
          {
            text: 'GitHub',
            link: 'https://github.com/1985312383/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/guide/': [
            {
              text: 'Chapter 1: Medical Imaging Basics',
              collapsed: false,
              items: [
                {
                  text: '1.1 Common Imaging Modality Principles',
                  collapsed: false,
                  items: [
                    { text: '1.1.1 CT', link: '/guide/ch01/01-modalities/01-ct' },
                    { text: '1.1.2 MRI', link: '/guide/ch01/01-modalities/02-mri' },
                    { text: '1.1.3 X-ray', link: '/guide/ch01/01-modalities/03-xray' },
                    { text: '1.1.4 PET/US', link: '/guide/ch01/01-modalities/04-pet-us' }
                  ]
                }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/1985312383/med-imaging-primer' }
        ],
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2025-present Your Name'
        }
      }
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      title: '医学影像处理开源教程',
      description: '从物理成像原理到深度学习的系统性入门指南',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '教程', link: '/zh/guide/ch01/01-modalities/01-ct' },
          {
            text: 'GitHub',
            link: 'https://github.com/1985312383/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/zh/guide/': [
            {
              text: '第1章 医学影像基础',
              collapsed: false,
              items: [
                {
                  text: '1.1 常见成像模态原理与特点',
                  collapsed: false,
                  items: [
                    { text: '1.1.1 CT', link: '/zh/guide/ch01/01-modalities/01-ct' },
                    { text: '1.1.2 MRI', link: '/zh/guide/ch01/01-modalities/02-mri' },
                    { text: '1.1.3 X-ray', link: '/zh/guide/ch01/01-modalities/03-xray' },
                    { text: '1.1.4 PET/US', link: '/zh/guide/ch01/01-modalities/04-pet-us' }
                  ]
                }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/1985312383/med-imaging-primer' }
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
    lineNumbers: true
  }
})