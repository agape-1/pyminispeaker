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
class Track:
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
            "The internal Event() used to alert when a track has finished.",
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
        """Pauses the track. Does nothing if the track is already paused."""
        self.paused = True

    def cont(self):
        """Unpauses the track. Does nothing if the track is already playing."""
        self.paused = False

    def mute(self):
        """Mutes the track. The audio will still be playing but it won't be heard. Does nothing if the track is already muted."""
        self.muted = True

    def unmute(self):
        """Unmutes the track. Does nothing if the track is not muted."""
        self.muted = False

    def wait(self):
        if not isinstance(self._signal, Event):
            raise TypeError(
                f"{self} is not a valid signal event instance."
            )
        return self._signal.wait()

    def assert_stream(self):
        """Ensures the Track has an available stream.

        Raises:
            ValueError: Underlying audio stream does not exist.
        """
        if self._stream is None:
            raise ValueError(
                f"{self} does not have a audio stream instance."
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