# Chat Completions

This example will show how to play back text to speech from a Chat Completions API via `pyminispeaker`.

## Instructions

1. Install a Chat Completions client

We recommend using [`openai`](https://pypi.org/project/openai/):

```sh
pip install openai
``` 

2. In your script, call and obtain any function that allows you to loop over raw audio data. We will be using the following code snippet from OpenAI's [`Audio API`](https://platform.openai.com/docs/guides/text-to-speech) documentation:


```python 
from openai import OpenAI

API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=API_KEY)
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input="Today is a wonderful day to build something people love!",
    instructions="Speak in a cheerful and positive tone."
) as response:
    audio_stream = response.iter_bytes(chunk_size=256)
```

3. Identify and match the sample rate and sample format of the audio stream.

Since we know their [`PCM`](https://platform.openai.com/docs/guides/text-to-speech#supported-output-formats) `response_format` has a sample rate of `24000` and `SIGNED16`, we can assign these like so:
```python
from miniaudio import SampleFormat
SAMPLE_RATE = 24000
SAMPLE_FORMAT = SampleFormat.SIGNED16
```

> [!WARNING]  
> Selecting the incorrect sample rate and format corresponding to the audio stream may result in non-working audio playback.

4. Pass the audio stream into `Speaker` with `SAMPLE_RATE` and `SAMPLE_FORMAT`.
```python
from openai import OpenAI
from minispeaker import Speakers
from miniaudio import SampleFormat

SAMPLE_RATE = 24000
SAMPLE_FORMAT = SampleFormat.SIGNED16

API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=API_KEY)
speaker = Speakers(sample_rate=SAMPLE_RATE, sample_format=SAMPLE_FORMAT)
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="coral",
    input="Today is a wonderful day to build something people love!",
    response_format="pcm", # <--- NOTE: `pcm` response format is selected to ensure a matching sample rate and sample format. 
    instructions="Speak in a cheerful and positive tone."
) as response:
    audio_stream = response.iter_bytes(chunk_size=256)
    speaker.play(audio_stream, name="My first TTS request") # Play the TTS response
    speaker["My first TTS request"].wait() # Wait for the audio to finish playing
```

## Asychronous Version

```python
from openai import AsyncOpenAI
from minispeaker import Speakers
from miniaudio import SampleFormat
import asyncio

SAMPLE_RATE = 24000
SAMPLE_FORMAT = SampleFormat.SIGNED16

API_KEY = "YOUR_API_KEY"
client = AsyncOpenAI(api_key=API_KEY)

async def TTS():
    speaker = Speakers(sample_rate=SAMPLE_RATE, sample_format=SAMPLE_FORMAT)
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input="Today is a wonderful day to build something people love!",
        response_format="pcm",
        instructions="Speak in a cheerful and positive tone."
    ) as response:
        audio_stream = response.iter_bytes(chunk_size=256)
        speaker.play(audio_stream, name="My first TTS request") # Play the TTS response
        await speaker["My first TTS request"].wait() # Wait for the audio to finish playing

asyncio.run(TTS())
```
