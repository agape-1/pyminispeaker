# Typing
from __future__ import annotations
from dataclasses import dataclass, InitVar
from typing_extensions import Annotated, Dict
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
            Event,
            "The internal Event() used to alert when a track has finished.",
        ]
    ] = None
    _stream: InitVar[
        Annotated[
            PlaybackCallbackGeneratorType,
            "The internal audio stream used to obtain the latest audio chunk.",
        ]
    ] = None

    def __post_init__(
        self, _signal: Event | None, _stream: PlaybackCallbackGeneratorType | None
    ):
        """Automatically assigns the internal private fields."""
        self._signal = _signal
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
        return self._signal.wait()

    def chunk(self, num_frames: int) -> ndarray:
        """Retrieves the latest audio chunk.

        Arg
            num_frames (int): The number of frames per chunk.

        Raises:
            ValueError: Underlying audio stream does not exist.

        Returns:
            ndarray: A audio chunk represented as a numpy array.
        """
        return asarray(self._stream.send(num_frames))
    
    
class TrackMapping(Dict[str, Track]):
    """Container for Track access and control
    """
    def clear(self):
        """Removes all current tracks. An alert is sent indicating all the tracks are finished.
        """
        for track in self.values():
            track._signal.set()
        super().clear()

    def __delitem__(self, name: str):
        """Remove a `Track` called `name`. An alert is sent indicating that`Track` is finished.

        Args:
            name (str): Name of the `Track`
        """
        self.__getitem__(name)._signal.set()
        super().__delitem__(name)