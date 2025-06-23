# Typing
from __future__ import annotations
from typing_extensions import List
from miniaudio import MiniaudioError, PlaybackDevice
from enum import Enum
from minispeaker.asyncsync import Event

# Main dependencies
from threading import Lock
from _miniaudio import ffi, lib # ffi, lib may be subject to change because it is imported from an internal library
from miniaudio import Devices

def default_speaker() -> str:
    """
    Retrieves the default speaker.

    Internally implemented using pyminiaudio's internal library, which is subject to change.

    Raises:
        MiniaudioError: If speaker enumeration fails

    Returns:
        str: The name of the default speaker.
    """
    """
    Modified from pyminiaudio's Devices.get_playbacks() and uses their internal lib and ffi implementation from
    https://github.com/irmen/pyminiaudio/blob/7bd6b529cd623359fa1009b9851e4482adc5e5bc/miniaudio.py.
    
    Since pyminiaudio is a Python wrapper for the C miniaudio library, default_speaker() uses CFFI to call C code,
    to list speaker devices checking if any one of them is marked as "default".
    """
    devices = Devices()

    # Initializing an empty structure to hold information about each speaker properties called 'playback_infos', 
    # and a integer to hold the total number of speakers found called 'playback_count'
    with ffi.new("ma_device_info**") as playback_infos, ffi.new("ma_uint32*") as playback_count:

        # Call this function to then 'fill' the information into 'playback_infos' and 'playback_count'
        result = lib.ma_context_get_devices(devices._context, playback_infos, playback_count, ffi.NULL,  ffi.NULL)
        if result != lib.MA_SUCCESS:
            raise MiniaudioError("cannot get device infos", result)
        
        # Parse this filled information to take the default speaker's name
        total_speakers = playback_count[0]
        for index in range(total_speakers):
            speaker = playback_infos[0][index]
            if speaker.isDefault: # Make the assumption that one of them will always be the default.
                name = ffi.string(speaker.name).decode()
                return name

def available_speakers() -> List[str]:
    """Retrieves all available speakers by name

    Returns:
        List[str]: A list of available speakers
    """
    return list(map(lambda speaker: speaker["name"], Devices().get_playbacks()))

class MaDeviceState(Enum):
    UNINITIALIZED= lib.ma_device_state_uninitialized
    STOPPED = lib.ma_device_state_stopped
    STARTED = lib.ma_device_state_started
    STOPPING = lib.ma_device_state_stopping
    STARTING = lib.ma_device_state_starting

class ConcurrentPlaybackDevice(PlaybackDevice):
    """
    Modified miniaudio `PlaybackDevice` class with accessible `ma_device_state` and
    thread-safe concurrency."""
    
    def __init__(self, *args, stopped: Event, **kwargs):
        self._stopped = stopped
        self._lock = Lock()
        super().__init__(*args, **kwargs)

    @property
    def state(self):
        """
        Retrieves `ma_device_state` from `PlaybackDevice`
        """
        if not self.closed:
            return MaDeviceState.UNINITIALIZED
        return MaDeviceState(lib.ma_device_is_started(self._device))

    @property 
    def volume(self) -> float | None:
        """
        Returns:
            float | None: _description_
        """
        if self.closed:
            return None
        return self._device.masterVolumeFactor

    @volume.setter
    def volume(self, vol: float):
        """
        Attempts to internally sets the volume of the internal PlaybackDevice.

        Uses internal implementation _device.masterVolumeFactor, so this function
        may not work if harmful changes are made to miniaudio or pyminiaudio.

        Function is no-op if it does not dynamically pass sanity checks on internal implementation.

        Args:
            vol (float): The initial volume of the speaker as a percent decimal.
        """
        # From https://www.reddit.com/r/miniaudio/comments/17vi68d/comment/kf8l3lw/
        if not self.closed and isinstance(self._device.masterVolumeFactor, float):
            self._device.masterVolumeFactor = vol

    def wait(self):
        """Waits for the `PlaybackDevice` to be stopped.
        """
        return self._stopped.wait()

    @property
    def starting(self):
        return self.state == MaDeviceState.STARTING

    @property
    def stopping(self):
        return self.state == MaDeviceState.STOPPING

    @property
    def stopped(self):
        return self.state == MaDeviceState.STOPPED

    @property
    def closed(self):
        "Is the `PlaybackDevice` uninitialized?"
        return not self._device # `self._device is maintained as None when the device is uninitialized at https://github.com/irmen/pyminiaudio/blob/601d03ceb6f7c3886aa295d0b4459424732f1547/miniaudio.py#L1435

    @property
    def started(self):
        return self.state == MaDeviceState.STARTED

    def start(self, callback_generator):
        with self._lock:
            if not self.closed and not self.starting and not self.started:
                self.volume = 1.0 # Hardcode volume factor to ensure consistent baseline volume
                super().start(callback_generator)
                self._stopped.clear()

    def stop(self):
        with self._lock:
            if not self.closed and not self.stopping and not self.stopped:
                super().stop()
                self._stopped.set()