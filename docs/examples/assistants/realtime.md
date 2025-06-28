# OpenAI Realtime Assistant

> [!NOTE]
> A guide is work in progress for enabling interruptions for the Realtime Assistant.

We are going to create a simple example for communicating with the Assistant. This application will playback responses from text input.

1. Install the required packages

```sh
pip install openai[realtime] aioconsole
```

2. We will be using an modified script from [`azure_realtime.py`](https://github.com/openai/openai-python/blob/main/examples/realtime/azure_realtime.py)

```python
import asyncio
import aioconsole
from openai import AsyncOpenAI

API_KEY='YOUR_API_KEY'

async def Assistant():
    client = AsyncOpenAI(api_key=API_KEY)
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={'modalities': ['text', 'audio']})

        while True:
            user_input = await aioconsole.ainput("Enter a message: ")
            if user_input == "q":
                break

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )
            await connection.response.create()
            async for event in connection:
                if event.type == "response.audio.done":
                    break
                elif event.type == "response.audio.delta":
                    pass
asyncio.run(Assistant())
```

3. Identify the corresponding sample rate and sample format of the audio data.

According to their [`Realtime API Documentation`](https://platform.openai.com/docs/api-reference/realtime-sessions/create), you can specify the audio data [`output_audio_format`](https://platform.openai.com/docs/api-reference/realtime-sessions/create#realtime-sessions-create-output_audio_format) to be `PCM` with 24kHZ and 16 bit-depth. We shall adjust our and configure these settings:

```python
import asyncio
import aioconsole
from openai import AsyncOpenAI
from miniaudio import SampleFormat

API_KEY='YOUR_API_KEY'

SAMPLE_RATE = 24000
SAMPLE_FORMAT = SampleFormat.SIGNED16

async def Assistant():
    client = AsyncOpenAI(api_key=API_KEY)
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={'modalities': ['text', 'audio'], 'output_audio_format': 'pcm16'})

        while True:
            user_input = await aioconsole.ainput("Enter a message: ")
            if user_input == "q":
                break

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )
            await connection.response.create()
            async for event in connection:
                if event.type == "response.audio.done":
                    break
                elif event.type == "response.audio.delta":
                    pass
asyncio.run(Assistant())
```

4. Create a custom generator to process the audio data.

The raw audio data is encoded in [`base64`](https://platform.openai.com/docs/api-reference/realtime-server-events/response/audio/delta):

```python
import asyncio
import aioconsole
from openai import AsyncOpenAI
from miniaudio import SampleFormat
from base64 import b64decode

API_KEY='YOUR_API_KEY'

SAMPLE_RATE = 24000
SAMPLE_FORMAT = SampleFormat.SIGNED16

async def process_audio(connection):
    async for event in connection:  
        if event.type == "response.audio.delta":
            audio = b64decode(event.delta)
            yield audio

async def Assistant():
    client = AsyncOpenAI(api_key=API_KEY)
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={'modalities': ['text', 'audio'], 'output_audio_format': 'pcm16'})

        while True:
            user_input = await aioconsole.ainput("Enter a message: ")
            if user_input == "q":
                break

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )
            await connection.response.create()
asyncio.run(Assistant())
```

5. Plug the audio stream into the `Speakers` with the corresponding sample format and sample rate:

```python
import asyncio
import aioconsole
from openai import AsyncOpenAI
from miniaudio import SampleFormat
from minispeaker import Speakers
from base64 import b64decode

API_KEY='YOUR_API_KEY'

SAMPLE_RATE = 24000
SAMPLE_FORMAT = SampleFormat.SIGNED16

async def process_audio(connection):
    async for event in connection:  
        if event.type == "response.audio.delta":
            audio = b64decode(event.delta)
            yield audio

async def Assistant():
    client = AsyncOpenAI(api_key=API_KEY)
    speakers = Speakers(sample_rate=SAMPLE_RATE, sample_format=SAMPLE_FORMAT)
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        speakers.play(process_audio(connection), name="speech")
        await connection.session.update(session={'modalities': ['text', 'audio'], 'output_audio_format': 'pcm16'})

        while True:
            user_input = await aioconsole.ainput("Enter a message: ")
            if user_input == "q":
                break

            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_input}],
                }
            )
            await connection.response.create()
asyncio.run(Assistant())
```

> [!WARNING]  
> Selecting the incorrect sample rate and format  may result in non-working audio playback.