import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Quila',
  description: 'QualiaTrace Language Model - Novel Dynamic Graph Reasoning System',
  head: [['link', { rel: 'icon', href: '/logo.svg' }]],
  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/' },
      { text: 'API', link: '/api/' },
      { text: 'GitHub', link: 'https://github.com/rand0mdevel0per/sxlm' }
    ],
    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'What is Quila?', link: '/guide/' },
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Architecture', link: '/guide/architecture' }
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'REST API', link: '/api/rest' },
          { text: 'WebSocket', link: '/api/websocket' },
          { text: 'Python Bindings', link: '/api/python' }
        ]
      },
      {
        text: 'Deployment',
        items: [
          { text: 'GCP Deployment', link: '/deployment/gcp' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/rand0mdevel0per/sxlm' }
    ]
  }
})
