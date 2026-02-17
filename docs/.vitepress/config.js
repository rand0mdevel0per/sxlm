export default {
  title: 'Qualia',
  description: 'AI assistant with 128M context and Plan-Think-Execute reasoning',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'API', link: '/api/' },
      { text: 'Architecture', link: '/architecture/' },
      { text: 'GitHub', link: 'https://github.com/rand0mdevel0per/sxlm' }
    ],
    sidebar: {
      '/api/': [
        { text: 'Quick Start', link: '/api/' },
        { text: 'Authentication', link: '/api/auth' },
        { text: 'Streaming', link: '/api/streaming' },
        { text: 'Python Client', link: '/api/python-client' }
      ],
      '/architecture/': [
        { text: 'Overview', link: '/architecture/' },
        { text: 'PTE Flow', link: '/architecture/pte' },
        { text: 'Ring Buffer', link: '/architecture/ring-buffer' },
        { text: 'el-trace', link: '/architecture/el-trace' },
        { text: 'KFE Storage', link: '/architecture/kfe' }
      ]
    }
  }
}
