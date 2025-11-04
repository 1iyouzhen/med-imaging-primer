import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import './custom.css'
import '@theojs/lumen/style'
import Mermaid from './components/Mermaid.vue'
import type { Theme } from 'vitepress'

export default {
    extends: DefaultTheme,
    Layout,
    enhanceApp({ app }) {
        // 注册全局 Mermaid 组件
        app.component('Mermaid', Mermaid)
    }
} satisfies Theme