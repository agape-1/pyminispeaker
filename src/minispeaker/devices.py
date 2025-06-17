# Typing
from typing_extensions import List
from miniaudio import MiniaudioError

# Main dependencies
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