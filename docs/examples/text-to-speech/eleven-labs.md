# ElevenLabs

This example will show how to play  text to speech via the Eleven Labs library and `minispeaker`.

## Instructions

1. Install [`ElevenLabs`](https://github.com/elevenlabs/elevenlabs-python):

```sh
pip install elevenlabs
``` 

2. In your script, call and obtain any function that allows you to loop over raw audio data. We will use the audio stream testing snippet from ElevenLab's [`README.md`](https://github.com/elevenlabs/elevenlabs-python?tab=readme-ov-file#streaming)


```python 
from elevenlabs import stream
from elevenlabs.client import ElevenLabs

API_KEY = "YOUR_API_KEY"
client = ElevenLabs()

text = "This is a test"
audio_stream = client.text_to_speech.stream(
    text=text,
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2"
)
```

3. Identify and match the sample rate and sample format of the audio stream.

Since ElevenLab's [`PCM`](https://elevenlabs.io/docs/capabilities/text-to-dialogue#supported-formats) supports a bit depth of 16, we will be using a Sample Format of `SampleFormat.SIGNED16`. Additionally their specifications allows for a wide variety different sample rates, allowing us to request audio stream with an identical sample rate of our default local playback device:
```python
from miniaudio import SampleFormat
SAMPLE_FORMAT = SampleFormat.SIGNED16  #  pcm `output_format` is corresponds to signed 16 sample format
speakers = Speakers(sample_format=SAMPLE_FORMAT)
SAMPLE_RATE = speakers.sample_rate
```

> [!WARNING]  
> Selecting the incorrect sample rate and format may result in non-working audio playback.

4. Pass the audio stream into `Speaker`.
```python
from elevenlabs.client import ElevenLabs
from minispeaker import Speakers
from miniaudio import SampleFormat


API_KEY = "YOUR_API_KEY"
SAMPLE_FORMAT = SampleFormat.SIGNED16  #  pcm `output_format` is corresponds to signed 16 sample format
client = ElevenLabs(api_key=API_KEY)
speakers = Speakers(sample_format=SAMPLE_FORMAT)
SAMPLE_RATE = speakers.sample_rate

text = "This is a test"
audio = client.text_to_speech.stream(
    text=text,
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format=f"pcm_{SAMPLE_RATE}",  # <-- Attempts to match sample rate to local playback device
)

speakers.play(audio, name=text)
speakers.wait()
```

> [!TIP]
> If the default `SAMPLE_RATE` is under API tier restrictions, you can directly lower`output_format`'s sample rate.

## Asychronous Version

```python
import asyncio
from elevenlabs.client import AsyncElevenLabs
from minispeaker import Speakers
from miniaudio import SampleFormat

text = "This is a test"
API_KEY = "YOUR_API_KEY"
SAMPLE_FORMAT = SampleFormat.SIGNED16  #  pcm `output_format` is corresponds to signed 16 sample format


async def TTS(text: str):
    eleven = AsyncElevenLabs(api_key=API_KEY)
    speakers = Speakers(sample_format=SampleFormat.SIGNED16)
    SAMPLE_RATE = speakers.sample_rate
    audio = eleven.text_to_speech.stream(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format=f"pcm_{SAMPLE_RATE}",  # <-- Attempts to match sample rate to local playback device
    )

    speakers.play(audio, name=text)
    await speakers.wait()

asyncio.run(TTS(text))
```
