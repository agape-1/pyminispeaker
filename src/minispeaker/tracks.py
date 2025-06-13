# Typing
from __future__ import annotations
from dataclasses import dataclass, InitVar
from typing_extensions import Annotated, Optional
from numpy import ndarray
from miniaudio import PlaybackCallbackGeneratorType
from minispeaker.asyncsync import Event

# Main dependencies
from numpy import asarray





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