import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import './custom.css'
import '@theojs/lumen/style'
import Mermaid from './components/Mermaid.vue'
import type { Theme } from 'vitepress'
import 'viewerjs/dist/viewer.min.css';
import imageViewer from 'vitepress-plugin-image-viewer';
import vImageViewer from 'vitepress-plugin-image-viewer/lib/vImageViewer.vue';
import { useRoute } from 'vitepress';

export default {
    extends: DefaultTheme,
    Layout,
    enhanceApp({ app }) {
        // 注册全局 Mermaid 组件
        app.component('Mermaid', Mermaid)
        // 注册全局组件（可选）
        app.component('vImageViewer', vImageViewer);
    },
    setup() {
        const route = useRoute();
        // 启用插件
        imageViewer(route);
    },
} satisfies Theme
