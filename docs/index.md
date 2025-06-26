---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "pyminispeaker"
  #text: "Pythonic Speaker Library"
  tagline: "Pythonic audio playback through miniaudio"
  actions:
    - theme: brand
      text:  Quickstart
      link: /getting-started # TODO: Implement 'getting-started.md'
    - theme: alt
      text: Usage Examples
      link: /examples # TODO: Implement `examples.md`

features:
  - title: Volume Mixer
    details: Mute, pause, and alter the volume of different tracks with one line of code.
  - title: Python first
    details: Plug and play any python generator with a simple, intuitive interface
  - title: Asynchronous Support
    details: Concurrently wait for any audio track to finish
