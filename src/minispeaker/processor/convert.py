# Typing
from typing_extensions import Union
from numpy import uint8, int16, int32, float32

# Helpers
from warnings import warn
from miniaudio import SampleFormat


def sampleformat_to_dtype(
    sample_format: SampleFormat
) -> Union[uint8, int16, int32, float32]:
    """Converts a `SampleFormat` to the numpy equivalent `dtype`.

    Args:
        sample_format (SampleFormat): miniaudio `SampleFormat` of the audio sample.

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
    """Converts a numpy `dtype` to an equivalent `SampleFormat`.

    Args:
        dtype (Union[None, uint8, int16, int32, float32]): Numpy dtype of the audio sample.

    Raises:
        ValueError: When a inconvertible dtype is given

    Returns:
        SampleFormat: Corresponding miniaudio `SampleFormat`.
    """
    convert = {
        None: SampleFormat.UNKNOWN,
        uint8: SampleFormat.UNSIGNED8,
        int16: SampleFormat.SIGNED16,
        int32: SampleFormat.SIGNED32,
        float32: SampleFormat.FLOAT32,
    }
    if dtype not in convert:
        raise ValueError(f"No known {SampleFormat} is supported for {dtype}. The supported dtypes are {list(convert.keys())}")
    return convert[dtype]
