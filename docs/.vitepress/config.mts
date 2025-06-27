import { defineConfigWithTheme } from 'vitepress'
import { generateSidebar } from 'vitepress-sidebar'
import baseConfig from 'vitepress-carbon/config'
import type { ThemeConfig } from 'vitepress-carbon/config'

const vitePressSidebarOptions = {
  hyphenToSpace: true,
  capitalizeFirst: true,
  useFolderLinkFromIndexFile: true,
  useFolderTitleFromIndexFile: true,
  keepMarkdownSyntaxFromTitle: true,
  useTitleFromFileHeading: true,
  collapsed: true,
  documentRootPath: '.'// NOTE: Assumes currently working directory `docs/`
};

export default defineConfigWithTheme<ThemeConfig>({
  extends: baseConfig,
  lang: 'en-US',
  base: '/pyminispeaker/',
  title: "pyminispeaker",
  description: "[`py`[thonic]`mini`audio](https://github.com/irmen/pyminiaudio) `speaker` abstraction library",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [ // TODO: Resolve navigation header
      { text: 'API Documentation', link: '/api-documentation' }, // TODO: Configure links for these invalid navigation items
      { text: 'Quickstart', link: '/getting-started/00-start' },
      { text: 'Examples', link: '/examples/text-to-speech/chat-completions' }
    ],
    sidebar: generateSidebar(vitePressSidebarOptions),
    search: {
      provider: 'local'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/agape-1/pyminispeaker' }
    ]
  }
})