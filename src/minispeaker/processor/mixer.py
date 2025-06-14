# Typing
from typing_extensions import List
from numpy import ndarray
from miniaudio import PlaybackCallbackGeneratorType

# Main dependencies
import numpy as np

def master_mixer(speaker) -> PlaybackCallbackGeneratorType:
    from minispeaker.player import Speakers
    """
    Audio processor that merges multiple audio streams together as a single generator, with master speaker mute, pause, and volume support.

    Args:
        speaker (Speaker): 
    Returns:
        PlaybackCallbackGeneratorType: Generator that supports miniaudio's playback callback

    Yields:
        Iterator[array]: A audio chunk
    """
    self: Speakers = speaker # TODO: Decouple main audio processor into different generators with more explicit argument passthrough
    num_frames = yield b""
    while not self._quit.is_set():
        if not self.paused:
            chunks: List[ndarray] = []
            volumes: List[float] = []
            for track in list(self.tracks.values()):
                if not track.paused and track._stream:
                    try:
                        chunks.append(track.chunk(num_frames))
                        volumes.append(track.volume)
                    except StopIteration:
                        continue
            if chunks and not self.muted and self.volume:
                audio = (
                    self.volume * np.average(chunks, axis=0, weights=volumes)
                ).astype(self._dtype)
                yield audio
            else:
                yield 0
    self._finished.set()