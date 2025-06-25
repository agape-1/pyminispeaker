import {defineConfig} from '@lando/vitepress-theme-default-plus/config';
import { generateSidebar } from 'vitepress-sidebar'

const vitePressSidebarOptions = {
  documentRootPath: '.'// NOTE: Assumes currently working directory `docs/`
};

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "pyminispeaker",
  description: "[`py`[thonic]`mini`audio](https://github.com/irmen/pyminiaudio) `speaker` abstraction library",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [ // TODO: Resolve navigation header
    ],
    multiVersionBuild: { // TODO: Resolve multi version build
      build: 'stable',
      match: '*',
      base: '/v/',
      satisfies: '*'
    },
    sidebar: generateSidebar(vitePressSidebarOptions),

    socialLinks: [
      { icon: 'github', link: 'https://github.com/agape-1/pyminispeaker' }
    ]
  }
})
