# Basics

Welcome!

To get started, first install the package:

```python
pip install minispeaker
```

## Playing a Sound File
Playing a sound file is very simple. Instantiate a `Speaker` object and
pass in any valid string `Path` of the file:

```python
from minispeaker import Speakers
sound_file = "myfile.mp3"
speaker = Speakers()
speaker.play(sound_file)
speaker[sound_file].wait()
```

## Dynamic Content
You can pass through any `AsyncIterator`, `Iterator`, `AsyncGenerator`, or `Generator` object(basically any object that you can loop into) that gives an audio data as any loose form of `array` or `bytes`.

```python
from minispeaker import Speakers


def sound_byte():
    return [b'\x00\x00\x0f\x00\x1f\x00/\x00>\x00N\x00]\x00m\x00}\x00\x8c\x00\x9c\x00\xac\x00\xbb\x00\xcb\x00\xda\x00\xea\x00\xfa\x00\t\x01\x19\x01(\x018\x01G\x01W\x01f\x01u\x01\x85\x01\x94\x01\xa4\x01\xb3\x01\xc2\x01\xd1\x01\xe1\x01\xf0\x01\xff\x01\x0e\x02\x1d\x02,\x02;\x02J\x02Y\x02h\x02w\x02\x86\x02\x95\x02\xa4\x02\xb2\x02\xc1\x02\xd0\x02\xde\x02\xed\x02'] * 100


speaker = Speakers()
speaker.play(iter(sound_byte()), name="sound_byte")
speaker["sound_byte"].wait()
```

## Track Control
Each individual `Track`'s volume, mute, pause, can be configured by accessing the Track's `name`:

```python
from time import sleep

speakers['sound_byte'].volume = 0.50  # 50 % volume
speakers['sound_byte'].pause()
sleep(0.5)
speakers['sound_byte'].cont()  # Stop pause
sleep(0.5)
speakers['sound_byte'].mute()
sleep(0.5)
speakers['sound_byte'].unmute()
```

You can assign a custom identity to your `Track`:

```python
KEY = "hashable"
speakers.play("sound_file.mp3", name=KEY)
speakers[KEY]  # Access Track
```

When a `Track` is paused, no attempt will be made to obtain the next audio byte. When a `Track` is muted, the audio will still be retrieved in the background, but it won't be played. You can have multiple sound files playing simultaneously all within the same `Speaker` instance.

## Track Signaling
Calling `wait` on any individual `Track` will be blocking until the the `Track` is finished:

```python
speakers['sound_byte'].wait()

await speakers['sound_byte'].wait()  # Asynchronously wait
```

## Master Speaker Override

Each `Speaker` instance will have their own equivalent features for wait, pausing, volume, and muting. These settings can be used to temporarily apply a global override to any `Track`s under the instance.

A `paused` Speaker results in no audio data fetch for any `Track`, even when they are individually unpaused. Similarly a muted `Speaker` will play nothing, regardless if everything was unpaused.

Calling `wait` on any `Speaker` instance will be blocking until the first very point where there are no `Track`s left to play:

```python
speakers.wait()
```
