# Typing
from __future__ import annotations
from dataclasses import dataclass, field
from asyncio import AbstractEventLoop
from typing_extensions import Callable, Annotated, Optional, Generator, AsyncGenerator
from collections.abc import AsyncIterator, Iterator
from types import AsyncGeneratorType, GeneratorType
from numpy.typing import ArrayLike, DTypeLike
from miniaudio import SampleFormat, PlaybackCallbackGeneratorType

# Helpers
from warnings import warn
from threading import Thread
from asyncio import get_event_loop, set_event_loop
from minispeaker.processor.convert import sampleformat_to_dtype
from minispeaker.asyncsync import Event, poll_async_generator
from minispeaker.tracks import TrackMapping
from inspect import getgeneratorstate, GEN_CREATED
from functools import partial
from atexit import register

# Main dependencies
from minispeaker.devices import default_speaker, ConcurrentPlaybackDevice
from minispeaker.tracks import Track
from minispeaker.processor.mixer import master_mixer
from minispeaker.processor.pipes import AudioPipeline, stream_sentinel, stream_handle_mute, stream_numpy_pcm_memory, stream_async_buffer, stream_bytes_to_array, stream_match_audio_channels, stream_num_frames, stream_pad, stream_as_generator, stream_async_as_generator
from miniaudio import (
    Devices,
    stream_with_callbacks
)
import numpy as np

@dataclass
class Speakers:
    """
    Class that offers an easy interface to play audio.

    Due to the supporting library implementation, each physical playback device should correspond to one Speaker class per Python process.
    In other words, don't try to have two Speakers with one device and expect functionality.

    Attributes:
        name (Optional[str]): The name of the speaker to playback to, found by available_speakers(). If no name is given, use the default speakers on the system.
        sample_rate (int): The sample rate of the audio in Hz.
        sample_format (Literal[SampleFormat.UNSIGNED8, SampleFormat.SIGNED16, SampleFormat.SIGNED24, SampleFormat.SIGNED32, SampleFormat.FLOAT32]: The bit depth of the audio. Defaults to SampleFormat.SIGNED16.
        channels (int): The number of audio channels.
        buffer_size (int): The size of each audio buffer in samples.
        volume (float): The initial volume of the speaker as a percent decimal.
    """

    name: Optional[str] = field(default_factory=default_speaker)
    sample_rate: int = 44100
    sample_format: SampleFormat = SampleFormat.SIGNED16
    channels: int = 1
    buffer_size: int = 128
    volume: float = 1.0

    def __post_init__(self):
        self._PlaybackDevice = ConcurrentPlaybackDevice(
            output_format=self.sample_format,
            nchannels=self.channels,
            sample_rate=self.sample_rate,
            device_id=self._speaker_name_to_id(self.name),
            stopped=Event(),
        )
        self.tracks = TrackMapping()
        self.paused = False
        self.muted = False
        register(self._on_exit)

    @property
    def _dtype(self) -> DTypeLike:
        """
        Returns:
            DTypeLike: Any `self`'s audio data dtype.
        """
        return sampleformat_to_dtype(self.sample_format)
    def _speaker_name_to_id(self, name: str) -> any:
        """Given a PlaybackDevice name, find the corresponding device_id.

        Args:
            name (str): The speaker name, found by available_speakers()

        Returns:
            any: A device_id for the speaker name
        """
        speakers = Devices().get_playbacks()
        return next(
            (speaker["id"] for speaker in speakers if speaker["name"] == name), None
        )

    def set_internal_volume(self, volume: float):
        """
        Attempts to internally sets the volume of the internal PlaybackDevice.

        Uses internal implementation _device.masterVolumeFactor, so this function
        may not work if harmful changes are made to miniaudio or pyminiaudio.

        Function is no-op if it does not dynamically pass sanity checks on internal implementation.

        Args:
            volume (float): The initial volume of the speaker as a percent decimal.
        """
        # From https://www.reddit.com/r/miniaudio/comments/17vi68d/comment/kf8l3lw/
        device = self._PlaybackDevice._device
        if isinstance(device.masterVolumeFactor, float):
            device.masterVolumeFactor = volume

    def pause(self):
        """Pauses the speaker. Does nothing if the speaker is already paused."""
        self.paused = True

    def cont(self):
        """Unpauses the speaker. Does nothing if the speaker is already playing."""
        self.paused = False

    def mute(self):
        """Mutes the speaker. The audio will still be playing but it won't be heard. Does nothing if the speaker is already muted."""
        self.muted = True

    def unmute(self):
        """Unmutes the speaker. Does nothing if the speaker is not muted."""
        self.muted = False

    def _handle_audio_end(self, name: str):
        """Tells anyone running Speakers().wait() to stop waiting on end of audio.

        Args:
            name (str): Name of the Track.
        """

        def alert_and_remove_track():
            del self.tracks[name] # NOTE: `TrackMapping()` assumed to handle signal waiting when deleted
            if not self.tracks:
                Thread(target=self._PlaybackDevice.stop, daemon=True).start()
        return alert_and_remove_track

    def _curried_audio_gen(self, audio: str | Generator[memoryview | bytes | ArrayLike, int, None] | AsyncGenerator[memoryview | bytes | ArrayLike, int], loop: AbstractEventLoop, track: Track) -> PlaybackCallbackGeneratorType:
        """Processes a variety of different audio formats by converting them to a synchronous generator.

        Args:
            audio (str | Generator[memoryview | bytes | ArrayLike, int, None] | AsyncGenerator[memoryview | bytes | ArrayLike, int]): Audio stream or audio file path.
            loop (AbstractEventLoop): Any loop.
            track (Track): The corresponding track of `audio`.

        Returns:
            PlaybackCallbackGeneratorType: A miniaudio compatible generator.

        Yields:
            ArrayLike: An audio chunk 
        """
        processor = AudioPipeline() # AudioPipeline chaining is AI-generated and modified for correctness
        if isinstance(audio, str):
            processor >>= (stream_numpy_pcm_memory, {
                "output_format": self.sample_format,
                "nchannels": self.channels,
                "sample_rate": self.sample_rate
        })
        elif isinstance(audio, AsyncGeneratorType):
            processor = (processor 
                >> (stream_async_buffer, {"max_buffer_chunks": 3})
                >> (poll_async_generator, {"loop": loop, "default_empty_factory": lambda: np.empty((0, self.channels))}))
        elif isinstance(audio, GeneratorType):
            if getgeneratorstate(audio) == GEN_CREATED:
                warn(f"Generator {audio} has not started. Please modify the generator to initially `yield b""`, or else the first audio chunk will skipped. Skipping the first audio chunk...")
                next(audio)
        processor = (processor
            >> (stream_bytes_to_array, self._dtype)
            >> (stream_match_audio_channels, self.channels)
            >> stream_num_frames
            >> (stream_pad, self.channels)
            >> (stream_handle_mute, track)
            >> (stream_with_callbacks, {"end_callback": self._handle_audio_end(track.name)}))
        return processor(audio)

    def _begin_playback(self, loop: AbstractEventLoop, audio: str | Generator[ArrayLike, int, None] | AsyncGenerator[ArrayLike, int], name: str):
        """Internal function to properly manipulate audio stream data for pause, mute, and wait functionality.

        Args:
            loop: (AbstractEventLoop): Any loop used to process asynchronous audio.
            audio (str | Generator[ArrayLike, int, None]): Audio stream or audio file path passed through from self.play()
            name (str): A custom name which will be accessible by self[name].
        """
        track = self.tracks[name]
        set_event_loop(loop)
        track._stream = audio = self._curried_audio_gen(audio, loop, track)

        mixer = master_mixer(tracks=self.tracks,
                                paused=lambda: self.paused, 
                                muted= lambda: self.muted,
                                volume=lambda: self.volume,
                                dtype=self._dtype,
                                stopped=self._PlaybackDevice._stopped)
        next(mixer)
        self._PlaybackDevice.start(mixer)

    def play(
        self,
        audio: str | Iterator[memoryview | bytes | ArrayLike] | AsyncIterator[memoryview | bytes| ArrayLike] | Generator[memoryview | bytes | ArrayLike, int, None] | AsyncGenerator[memoryview | bytes | ArrayLike, int],
        name: Annotated[Optional[str], "Custom name for the audio."] = None,
        volume: Annotated[Optional[float], "The initial track volume."] = None,
        paused: Annotated[Optional[bool], "Should the audio not play immediately?"] = False,
        muted: Annotated[Optional[bool], "Should the audio be muted immediately?"] = False,
        realtime: Annotated[Optional[bool], "Should the audio(if asynchronous) be played in realtime?"] = False
    ):
        """Plays audio to the speaker.

        Usage:
            speaker = Speakers(name="My device speakers")
            speaker.play("track.mp3")
            speaker.wait() # Wait until track is finished
            speaker.stop()

            or

            with Speaker(name="My device speakers") as speaker:
                speaker.play("track.mp3", 'special name')
                speaker.play("test.mp3") # Both track.mp3 and test.mp3 are playing
                speaker['special name'].wait() # Wait until 'special name', or 'track.mp3' is finished. 'test.mp3' might still be playing.
                speaker.wait() # Wait until all the tracks are finished

        Args:
            audio (str | Iterator[memoryview  |  bytes  |  ArrayLike] | AsyncIterator[memoryview  |  bytes |  ArrayLike] | Generator[memoryview  |  bytes  |  ArrayLike, int, None] | AsyncGenerator[memoryview  |  bytes  |  ArrayLike, int]): Audio file path or audio stream. The audio stream can either be any form of async/sync iterator or generator. Keep in mind that for generators, they must be pre-initialized via next() and yield audio chunks as some form of an array, to allow the ability to send() a number into the generator and receive a corresponding number of audio frames, instead of the unknown pre-set amount. See memory_stream() for an example.
            name (str): A custom name which will be accessible by self[name]. Defaults to `audio`.
            volume (float): The individual Track's volume. Defaults to `self.volume`.
            paused (bool): Should the audio be immediately paused before playback? Defaults to `False`.
            muted (bool): Should the audio be immediately muted before playback? Defaults to `False`.
            realtime (bool): Should the audio(if asynchronous) be played in realtime? Defaults to `False`.
        Raises:
            TypeError: The audio input is not valid and must be a correct file path.
        """
        if isinstance(audio, AsyncIterator):
            audio = stream_async_as_generator(audio)
        if isinstance(audio, Iterator):
            audio = stream_as_generator(audio)

        if not isinstance(audio, (str, GeneratorType, AsyncGeneratorType)):
            raise TypeError(f"{audio} is not a string, iterator, or a generator")

        if name is None:
            name = audio

        if volume is None:
            volume = self.volume

        track = Track(
            name=name, paused=paused, muted=muted, volume=volume, realtime=realtime, _signal=Event(), _stream=stream_sentinel()
        )
        self.tracks[name] = track

        process_audio = Thread(target=self._begin_playback, args=(get_event_loop(), audio, name), daemon=True)
        process_audio.start()

    @property
    def _on_exit(self) -> Callable[[], None]:
        """
        Internal function used to close the `Speaker` object.
    
        This implementation is a function factory disguised as a method via the `@property` decorator.
        In other words, every call to `Speaker._exit` will return an new instance of the `close()` function.
        
        The purpose behind the implementation is to prevent [memory collection holds](https://stackoverflow.com/questions/16333054/what-are-the-implications-of-registering-an-instance-method-with-atexit-in-pytho) on cleanup.

        Usage:
            speaker._on_exit()  # Exit now
            atexit.register(speaker._on_exit)  # Register for later
        """
        def close(stopped: Event, tracks: TrackMapping, PlaybackDevice: ConcurrentPlaybackDevice):
            """Release all resources and any signaling.
            Args:
                stopped (Event): Signal for when the Speaker is playing any track.
                tracks (TrackMapping): Dictionary of all tracks keyed by name.
                PlaybackDevice (PlaybackDevice): Internal `Speaker` playback device.
            """
            stopped.set()
            tracks.clear()
            PlaybackDevice.close()
        return partial(close, stopped=self._PlaybackDevice._stopped, tracks=self.tracks, PlaybackDevice=self._PlaybackDevice)

    def exit(self):
        """Close the speaker. After Speakers().exit() is called, any calls to play with this Speaker object will be undefined behavior."""
        self._on_exit()

    def wait(self):
        """By default, playing will run in the background while other code is executed.
        Call this function to wait for the speaker to finish playing before moving to the next part of the code.
        """
        return self._PlaybackDevice.wait()

    def clear(self):
        """Removes all current tracks. An alert is sent indicating all the tracks are finished.
        """
        self.tracks.clear()

    def __getitem__(self, key: str):
        """Helper to allow access to an individual Track via self['track']."""
        return self.tracks[key]

    def __delitem__(self, name: str):
        """Remove a `Track` called `name`. An alert is sent indicating that`Track` is finished.

        Args:
            name (str): Name of the `Track`
        """
        del self.tracks[name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()
