# Typing
from __future__ import annotations
from dataclasses import dataclass, InitVar, field
from asyncio import AbstractEventLoop
from typing_extensions import Annotated, Dict, List, Optional, Generator, Literal, AsyncGenerator
from types import AsyncGeneratorType, GeneratorType
from numpy import ndarray
from numpy.typing import ArrayLike, DTypeLike
from miniaudio import SampleFormat, PlaybackCallbackGeneratorType

# Helpers
from warnings import warn
from threading import Thread
from asyncio import get_event_loop, set_event_loop
from minispeaker.utils.miniaudios import sampleformat_to_dtype
from minispeaker.asyncsync import Event, poll_async_generator
from numpy import asarray
from inspect import getgeneratorstate, GEN_CREATED

# Main dependencies
from minispeaker.devices import default_speaker
from minispeaker.processor.pipes import stream_numpy_pcm_memory, stream_async_buffer, stream_bytes_to_array, stream_match_audio_channels, stream_num_frames
from miniaudio import (
    Devices,
    PlaybackDevice,
    stream_with_callbacks
)
import numpy as np


@dataclass
class Song:
    """
    Class that holds auxiliary information on a audio piece being played.
    """

    name: str
    paused: bool
    muted: bool
    volume: float
    realtime: bool

    # InitVar is used here to hide the internal fields.
    _signal: InitVar[
        Annotated[
            Optional[Event],
            "The internal Event() used to alert when a song has finished.",
        ]
    ] = None
    _stream: InitVar[
        Annotated[
            Optional[PlaybackCallbackGeneratorType],
            "The internal audio stream used to obtain the latest audio chunk.",
        ]
    ] = None

    def __post_init__(
        self, _signal: Event | None, _stream: PlaybackCallbackGeneratorType | None
    ):
        """Automatically assigns the internal private fields."""
        if _signal:
            self._signal = _signal
        if _stream:
            self._stream = _stream

    def pause(self):
        """Pauses the song. Does nothing if the song is already paused."""
        self.paused = True

    def cont(self):
        """Unpauses the song. Does nothing if the song is already playing."""
        self.paused = False

    def mute(self):
        """Mutes the song. The audio will still be playing but it won't be heard. Does nothing if the song is already muted."""
        self.muted = True

    def unmute(self):
        """Unmutes the song. Does nothing if the song is not muted."""
        self.muted = False

    def wait(self):
        if not isinstance(self._signal, Event):
            raise TypeError(
                f"speakers.py: {self} does not have an correct signal event instance."
            )
        return self._signal.wait()

    def assert_stream(self):
        """Ensures the Song has an available stream.

        Raises:
            ValueError: Underlying audio stream does not exist.
        """
        if self._stream is None:
            raise ValueError(
                f"speakers.py {self} does not have a audio stream instance."
            )

    def chunk(self, num_frames: int) -> ndarray:
        """Retrieves the latest audio chunk.

        Args:
            num_frames (int): The number of frames per chunk.

        Raises:
            ValueError: Underlying audio stream does not exist.

        Returns:
            ndarray: A audio chunk represented as a numpy array.
        """
        self.assert_stream()
        return asarray(self._stream.send(num_frames))

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
    sample_format: Literal[SampleFormat.UNSIGNED8,
            SampleFormat.SIGNED16,
            SampleFormat.SIGNED24,
            SampleFormat.SIGNED32,
            SampleFormat.FLOAT32] = SampleFormat.SIGNED16
    channels: int = 2
    buffer_size: int = 128
    volume: float = 1.0

    def __post_init__(self):
        self._PlaybackDevice = PlaybackDevice(
            output_format=self.sample_format,
            nchannels=self.channels,
            sample_rate=self.sample_rate,
            device_id=self._speaker_name_to_id(self.name),
        )
        self.set_internal_volume(1.0)
        self._quit = Event()
        self.songs: Dict[str, Song] = dict()
        self._finished = Event()
        self._playable = False
        self.paused = False
        self.muted = False

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

    def _handle_audio_end(self, name: str, song_end: Event):
        """Tells anyone running Speakers().wait() to stop waiting on end of audio.

        Args:
            name (str): Name of the Song.
            song_end (Event): Anyone running wait() on song_end Event.
        """

        def alert_and_remove_song():
            del self.songs[name]
            song_end.set()

        return alert_and_remove_song

    def processor_running(self) -> bool:
        """
        Returns:
            bool: Is the main audio processor running?
        """
        return self._playable

    def is_playable(self) -> bool:
        """
        Returns:
            bool: Does the speaker have any songs ready to be played?  
        """
        return len(self.songs) >= 1

    def _main_audio_processor(self) -> PlaybackCallbackGeneratorType:
        """
        Audio processor that merges multiple audio streams together as a single generator.
        Allows per-application volume.

        Returns:
            PlaybackCallbackGeneratorType: Generator that supports miniaudio's playback callback

        Yields:
            Iterator[array]: A audio chunk
        """

        def pad(chunk: ndarray) -> ndarray:
            """When calculating np.average to mix multiple audio streams,
            the function assumes all audio streams are identical in shape.
            This is the case until an audio stream is nearing its end, whereby
            it returns an trimmed audio stream. pad() ensures that the
            trimmed audio stream is padded.

            Args:
                chunk (ndarray): An audio chunk.

            Returns:
                ndarray: An audio chunk padded to the uniform shape.
            """
            if not (chunk.ndim and chunk.size):
                return np.zeros((num_frames, self.channels))
            if chunk.shape[0] != num_frames:
                padded = chunk.copy()
                padded.resize((num_frames, self.channels))
                return padded
            return chunk

        num_frames = yield b""
        while not self._quit.is_set() and self.is_playable():
            if not self.paused:
                chunks: List[ndarray] = []
                volumes: List[float] = []

                for song in list(self.songs.values()):
                    if not song.paused and song._stream:
                        try:
                            chunk = song.chunk(num_frames)
                        except StopIteration:
                            continue

                        if not song.muted:
                            chunks.append(chunk)
                            volumes.append(song.volume)

                if chunks and not self.muted and self.volume:
                    chunks = list(map(pad, chunks))
                    audio = (
                        self.volume * np.average(chunks, axis=0, weights=volumes)
                    ).astype(self._dtype)
                    yield audio
                else:
                    yield 0
        self._playable = False
        self._finished.set()

    def _unify_audio_types(self, audio: str | Generator[memoryview | bytes | ArrayLike, int, None] | AsyncGenerator[memoryview | bytes | ArrayLike, int], loop: AbstractEventLoop, song: Song) -> PlaybackCallbackGeneratorType:
        """Processes a variety of different audio formats by converting them to a synchronous generator.

        Args:
            audio (str | Generator[memoryview | bytes | ArrayLike, int, None] | AsyncGenerator[memoryview | bytes | ArrayLike, int]): Audio stream or audio file path.
            loop (AbstractEventLoop): Any loop.
            song (Song): The corresponding song of `audio`.

        Returns:
            PlaybackCallbackGeneratorType: A miniaudio compatible generator.

        Yields:
            ArrayLike: An audio chunk 
        """
        if isinstance(audio, str):
            audio = stream_numpy_pcm_memory(
                            filename=audio,
                            output_format=self.sample_format,
                            nchannels=self.channels,
                            sample_rate=self.sample_rate
                        )
            next(audio)
        elif isinstance(audio, AsyncGeneratorType):
            audio = stream_async_buffer(audio, max_buffer_chunks=3) # TODO: Make `max_buffer_chunks` accessible from higher level interface
            audio = poll_async_generator(audio,  loop=loop, default_empty_factory=lambda: np.empty((0, self.channels)))
            next(audio)
        elif isinstance(audio, GeneratorType):
            if getgeneratorstate(audio) == GEN_CREATED:
                warn(f"mic.py: The audio generator {audio} has not started. Please modify the generator to `yield b"" initially`, or else the first audio chunk will skipped. Skipping...")
                next(audio)
        audio = stream_bytes_to_array(audio, self._dtype)
        next(audio)
        audio = stream_match_audio_channels(audio, self.channels)
        next(audio)
        audio = stream_num_frames(audio)
        next(audio)
        return audio

    def _play(self, loop: AbstractEventLoop, audio: str | Generator[ArrayLike, int, None] | AsyncGenerator[ArrayLike, int], name: str):
        """Internal function to properly manipulate audio stream data for pause, mute, and wait functionality.

        Args:
            loop: (AbstractEventLoop): Any loop used to process asynchronous audio.
            audio (str | Generator[ArrayLike, int, None]): Audio stream or audio file path passed through from self.play()
            name (str): A custom name which will be accessible by self[name].
        Raises:
            TypeError: The audio input is not valid and must be a correct file path.
        """
        song = self.songs[name]
        end_signal = song._signal
        set_event_loop(loop)
        if not isinstance(audio, (str, GeneratorType, AsyncGeneratorType)):
            raise TypeError('speakers.py: audio is not a valid file path, nor is it a generator to stream audio chunks')
        audio = self._unify_audio_types(audio, loop, song)
        audio_controller = stream_with_callbacks(sample_stream=audio, end_callback=self._handle_audio_end(name, end_signal))
        next(audio_controller)

        song._stream = audio_controller

        if not self.processor_running():
            processor = self._main_audio_processor()
            next(processor)
            self._playable = True
            self._PlaybackDevice.start(processor)

        self._quit.clear()
        self._quit.tevent.wait() # This function is intended to be run as a separate thread, requiring tevent <--> threading.Event()

    def play(
        self,
        audio: str | Generator[memoryview | bytes | ArrayLike, int, None] | AsyncGenerator[memoryview | bytes | ArrayLike, int],
        name: Annotated[Optional[str], "Custom name for the audio."] = None,
        volume: Annotated[Optional[float], "The initial song volume."] = None,
        paused: Annotated[Optional[bool], "Should the audio not play immediately?"] = False,
        muted: Annotated[Optional[bool], "Should the audio be muted immediately?"] = False,
        realtime: Annotated[Optional[bool], "Should the audio(if asynchronous) be played in realtime?"] = False
    ):
        """Plays audio to the speaker.

        Usage:
            speaker = Speakers(name="My device speakers")
            speaker.play("song.mp3")
            speaker.wait() # Wait until song is finished
            speaker.stop()

            or

            with Speaker(name="My device speakers") as speaker:
                speaker.play("song.mp3", 'special name')
                speaker.play("test.mp3") # Both song.mp3 and test.mp3 are playing
                speaker['special name'].wait() # Wait until 'special name', or 'song.mp3' is finished. 'test.mp3' might still be playing.
                speaker.wait() # Wait until all the songs are finished

        Args:
            audio (str | Generator[memoryview | bytes | ArrayLike, int, None] | AsyncGenerator[memoryview | bytes | ArrayLike, int]): Audio file path or audio stream. The audio stream must be pre-initialized via next() and yield audio chunks as some form of an array. If you send() a number into the generator rather than just using next() on it, you'll get that given number of frames, instead of the default configured amount. See memory_stream() for an example.
            name (str): A custom name which will be accessible by self[name]. Defaults to `audio`.
            volume (float): The individual Song's volume. Defaults to `self.volume`.
            paused (bool): Should the audio be immediately paused before playback? Defaults to `False`.
            muted (bool): Should the audio be immediately muted before playback? Defaults to `False`.
            realtime (bool): Should the audio(if asynchronous) be played in realtime? Defaults to `False`.
        """
        if name is None:
            name = audio

        if volume is None:
            volume = self.volume

        self._finished.clear()

        song = Song(
            name=name, paused=paused, muted=muted, volume=volume, realtime=realtime, _signal=Event()
        )
        self.songs[name] = song

        play_background = Thread(target=self._play, args=(get_event_loop(), audio, name), daemon=True)
        play_background.start()

    def exit(self):
        """Close the speaker. After Speakers().exit() is called, any calls to play with this Speaker object will be undefined behavior."""
        self._PlaybackDevice.stop()
        self._quit.set()

    def wait(self):
        """By default, playing will run in the background while other code is executed.
        Call this function to wait for the speaker to finish playing before moving to the next part of the code.
        """
        return self._finished.wait()

    def clear(self):
        """Removes all current songs.
        """
        self.songs.clear()

    def __getitem__(self, key: str):
        """Helper to allow access to an individual Song via self['song']."""
        return self.songs[key]

    def __delitem__(self, key: str):
        """Helper to quickly remove an individual Song via del self['song']."""
        del self.songs[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()
