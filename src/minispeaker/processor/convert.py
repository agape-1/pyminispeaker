# Typing
from typing_extensions import Union
from numpy import uint8, int16, int32, float32

# Helpers
from warnings import warn
from miniaudio import SampleFormat


def sampleformat_to_dtype(
    sample_format: SampleFormat
) -> Union[uint8, int16, int32, float32]:
    """
    Args:
        sample_format (Literal[SampleFormat.UNKNOWN, SampleFormat.UNSIGNED8, SampleFormat.SIGNED16, SampleFormat.SIGNED24, SampleFormat.SIGNED32, SampleFormat.FLOAT32]): miniaudio `SampleFormat` of the audio sample.

    Returns:
        Union[uint8, int16, int32, float32]: A corresponding numpy dtype.
    """
    convert = {
        SampleFormat.UNKNOWN: None,
        SampleFormat.UNSIGNED8: uint8,
        SampleFormat.SIGNED16: int16,
        SampleFormat.SIGNED24: int32,
        SampleFormat.SIGNED32: int32,
        SampleFormat.FLOAT32: float32,
    }
    if (
        isinstance(sample_format, SampleFormat)
        and sample_format == SampleFormat.SIGNED24
    ):
        warn(
            f"Numpy arrays does not directly support the format {SampleFormat.SIGNED24}. Returning {convert[sample_format]}..."
        )
    return convert[sample_format]


def dtype_to_sampleformat(dtype: Union[None, uint8, int16, int32, float32]) -> SampleFormat:
    """
    Args:
        dtype (Union[None, uint8, int16, int32, float32]): Numpy dtype of the audio sample.

    Returns:
        Union[SampleFormat.UNKNOWN, SampleFormat.UNSIGNED8, SampleFormat.SIGNED16, SampleFormat.SIGNED24, SampleFormat.SIGNED32, SampleFormat.FLOAT32]: Corresponding miniaudio `SampleFormat`.
    """
    convert = {
        None: SampleFormat.UNKNOWN,
        uint8: SampleFormat.UNSIGNED8,
        int16: SampleFormat.SIGNED16,
        int32: SampleFormat.SIGNED32,
        float32: SampleFormat.FLOAT32,
    }
    return convert[dtype]
