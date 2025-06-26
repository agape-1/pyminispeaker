import {defineConfig} from '@lando/vitepress-theme-default-plus/config';
import { generateSidebar } from 'vitepress-sidebar'

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

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "pyminispeaker",
  description: "[`py`[thonic]`mini`audio](https://github.com/irmen/pyminiaudio) `speaker` abstraction library",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [ // TODO: Resolve navigation header
      { text: 'API Documentation', link: '/building/source' }, // TODO: Configure links for these invalid navigation items
      { text: 'Quickstart', link: '/building/source' },
      { text: 'Examples', link: '/building/source' }
    ],
    multiVersionBuild: { // TODO: Resolve multi version build
      build: 'stable',
      match: '*',
      base: '/v/',
      satisfies: '*'
    },
    sidebar: generateSidebar(vitePressSidebarOptions),
    search: {
      provider: 'local'
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/agape-1/pyminispeaker' }
    ]
  }
})
