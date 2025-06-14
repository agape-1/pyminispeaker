# Typing
from typing_extensions import List, Callable, Dict
from minispeaker.tracks import Track
from minispeaker.asyncsync import Event
from numpy.typing import DTypeLike
from numpy import ndarray
from miniaudio import PlaybackCallbackGeneratorType

# Main dependencies
import numpy as np

def master_mixer(
    tracks: Dict[str, Track],
    closed: Event,
    paused: Callable[[], bool],
    muted: Callable[[], bool],
    volume: Callable[[], float],
    dtype: DTypeLike,
    exited: Event
) -> PlaybackCallbackGeneratorType: # TODO: Reduce `closed` Event and `exited` Event to single event.
    """Audio processor that merges multiple audio stream with master controls.

    Args:
        tracks (Dict[str, Track]): Multiple audio streams represented as a dictionary of `Track`s.
        closed (Event): To signal when `Speaker` is closed.
        paused (Callable[[], bool]): When `paused` is evaluated to `True`, no audio streams will continue.
        muted (Callable[[], bool]): MWhen `muted` is evaluted to `True`, all audio streams will continue but will play no sound.
        volume (Callable[[], float]): Master volume.
        dtype (DTypeLike): SampleFormat equivalent of the underlying audio streams.
        exited (Event): To signal when `master_mixer` is finished.

    Yields:
        Iterator[PlaybackCallbackGeneratorType]: Miniaudio compatiable audio stream generator
    """
    num_frames = yield b""
    
    while not closed.is_set():
        if not paused():
            chunks: List[ndarray] = []
            volumes: List[float] = []

            for track in list(tracks.values()):
                if not track.paused and track._stream:
                    try:
                        chunks.append(track.chunk(num_frames))
                        volumes.append(track.volume)
                    except StopIteration:
                        continue
            if chunks and not muted():
                audio = (
                    volume() * np.average(chunks, axis=0, weights=volumes)
                ).astype(dtype)
                yield audio
            else:
                yield 0
    exited.set()