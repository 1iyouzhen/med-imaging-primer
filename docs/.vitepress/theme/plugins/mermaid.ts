import type MarkdownIt from 'markdown-it'

/**
 * Mermaid markdown-it 插件
 * 将 ```mermaid 代码块转换为 Vue 组件
 */
export function MermaidPlugin(md: MarkdownIt) {
  const fence = md.renderer.rules.fence!
  
  md.renderer.rules.fence = (...args) => {
    const [tokens, idx] = args
    const token = tokens[idx]
    const lang = token.info.trim()

    // 检查是否是 mermaid 代码块
    if (lang === 'mermaid') {
      const code = token.content.trim()
      
      // 转义代码中的特殊字符
      const escapedCode = code
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
      
      // 生成唯一 ID
      const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`
      
      // 返回 Vue 组件标签
      return `<Mermaid id="${id}" code="${escapedCode}" />`
    }

    // 其他代码块使用默认渲染
    return fence(...args)
  }
}

