# Typing
from __future__ import annotations
from typing_extensions import Concatenate, ParamSpec, Buffer, Generator, Union, Callable, AsyncGenerator, Any, TypeVar, Tuple, TypeAlias, Dict
from numpy import ndarray
from numpy.typing import ArrayLike, DTypeLike
from miniaudio import SampleFormat, DitherMode, FramesType

# Helpers
from asyncio import create_task, Queue
from minispeaker.asyncsync import Event, poll_async_generator

# Main dependencies
from minispeaker.tracks import Track
from miniaudio import (
    decode_file,
    stream_with_callbacks,
    PlaybackCallbackGeneratorType,
)
import numpy as np

def memory_stream(arr: ndarray) -> PlaybackCallbackGeneratorType:
    """Converts a numpy array into a stream.

    Args:
        arr (ndarray): An numpy array of shape(-1, channels).

    Returns:
        PlaybackCallbackGeneratorType: Generator that supports miniaudio's playback callback

    Yields:
        Iterator[ndarray]: A audio chunk represented as a numpy subarray.
    """
    # Modified from https://github.com/irmen/pyminiaudio/blob/7bd6b529cd623359fa1009b9851e4482adc5e5bc/examples/numpysample.py#L14
    required_frames = yield b""  # generator initialization
    frames = 0
    while frames < len(arr):
        frames_end = frames + required_frames
        required_frames = yield arr[frames:frames_end]
        frames = frames_end


def stream_numpy_pcm_memory(
    filename: str,
    output_format: SampleFormat = SampleFormat.SIGNED16,
    nchannels: int = 2,
    sample_rate: int = 44100,
    dither: DitherMode = DitherMode.NONE,
) -> PlaybackCallbackGeneratorType:
    """Convenience function that returns a generator to decode and stream any source of encoded audio data.
    Stream result is chunks of raw PCM samples as a numpy array.
    If you send() a number into the generator rather than just using next() on it, you'll get that given number of frames,
    instead of the default configured amount. This is particularly useful to plug this stream into an audio device callback
    that wants a variable number of frames per call.

    Args:
        filename (str): _description_
        output_format (SampleFormat, optional): _description_. Defaults to SampleFormat.SIGNED16.
        nchannels (int, optional): _description_. Defaults to 2.
        sample_rate (int, optional): _description_. Defaults to 44100.
        dither (DitherMode, optional): _description_. Defaults to DitherMode.NONE.

    Returns:
        PlaybackCallbackGeneratorType: _description_
    """
    # Modified from https://github.com/irmen/pyminiaudio/blob/7bd6b529cd623359fa1009b9851e4482adc5e5bc/examples/numpysample.py#L25
    audio = decode_file(
        filename=filename,
        output_format=output_format,
        nchannels=nchannels,
        sample_rate=sample_rate,
        dither=dither,
    )
    numpy_pcm = np.array(audio.samples, dtype=np.int16).reshape(-1, nchannels)
    return memory_stream(numpy_pcm)

def stream_async_with_callbacks(sample_stream: AsyncGenerator[bytes | ArrayLike, int],
                          progress_callback: Union[Callable[[int], None], None] = None,
                          frame_process_method: Union[Callable[[FramesType], FramesType], None] = None,
                          end_callback: Union[Callable, None] = None) -> PlaybackCallbackGeneratorType:
    """Convenience synchronous generator function to add callback and processing functionality, allowing synchronous playback from a asynchronous stream. You can specify :
    A callback function that gets called during play and takes an int for the number of frames played.
    A function that can be used to process raw data frames before they are yielded back (takes an array.array or bytes, returns an array.array or bytes) *Note: if the processing method is slow it will result in audio glitchiness
    """ 
    blank_audio_if_pending= poll_async_generator(sample_stream, default_empty_factory=lambda : b"")
    next(blank_audio_if_pending)
    return stream_with_callbacks(sample_stream=blank_audio_if_pending, progress_callback=progress_callback, frame_process_method=frame_process_method, end_callback=end_callback)

def stream_num_frames(sample_stream: Generator[ArrayLike, Any, None]) -> PlaybackCallbackGeneratorType:
    """Convenience generator function with dynamic audio buffer management to guarantee a certain audio chunk size per iteration. 
    If you send() a number into the generator rather than just using next() on it, you'll get that given number of frames, instead of the default configured amount. This is particularly useful to plug this stream into an audio device callback that wants a variable number of frames per call.

    Args:
        sample_stream (Generator[ArrayLike, Any, None]): Any ArrayLike generator.

    Returns:
        PlaybackCallbackGeneratorType: 

    Yields:
        ArrayLike: An audio chunk 
    """
    def send():
        return np.asarray(sample_stream.send(num_frames))
    
    def next():
        return np.asarray(sample_stream.__next__())
    num_frames = yield b""
    get = send if hasattr(sample_stream, "send") else next # Extra compatibility for iterator only audio.
    audio = get()
    while True:
        try:
            if len(audio) >= num_frames: # TODO: Find someway to add a 'safety' buffer of double its chunks for smooth playback.
                piece, audio = audio[:num_frames], audio[num_frames:]
                num_frames = yield piece
            else:
                more = get()
                audio = np.concatenate((audio.ravel(), more.ravel())).reshape((-1, audio.shape[1]))
        except StopIteration:
            yield audio[:min(len(audio), num_frames)] # Give out remaining audio, but never give audio more than it has requested
            break

def stream_match_audio_channels(sample_stream: Generator[ArrayLike, int, None],
                                channels: int) -> Generator[ndarray, int, None]:
    """Convenience generator function to automatically reformat any `sample_stream` data as a numpy channeled array.

    Args:
        sample_stream (Generator[ArrayLike, int, None]): Any ArrayLike generator.
        channels (int): _description_

    Yields:
        Generator[ndarray, int, None]: Audio data formatted with the correct `channels`.
    """
    num_frames = yield b""
    while True:
        try:
            audio = sample_stream.send(num_frames) #Modified from miniaudio.stream_with_callbacks()
            num_frames = yield np.asarray(audio).reshape((-1, channels))
        except StopIteration:
            break

async def stream_async_buffer(sample_stream: AsyncGenerator[ArrayLike, None], max_buffer_chunks: int) -> AsyncGenerator[ArrayLike, None]:
    """Asynchronous convenience generator function to prefetch audio for continuous playback. 

    Args:
        sample_stream (AsyncGenerator[ArrayLike, None]): Any asynchronous audio generator.
        max_buffer_chunks (int): The prefetched audio size will not exceed by this amount.
    Yields:        realtime (bool): Should the stream prioritize returning the most recent audio data over the later ones? Default to False.
        AsyncGenerator[ArrayLike, None]: Identical audio stream with buffer cache.
    """
    STREAM_FINISHED = None
    audio_ready = Event()
    async def background_stream():
        try:
            async for audio in sample_stream:
                await queue.put(audio)
                if not audio_ready.is_set() and queue.qsize() == 2: # When the audio queue first starts, buffer slightly to prevent choppiness on first chunk playback
                    audio_ready.set()
        finally:
            await queue.put(STREAM_FINISHED)

    queue = Queue(maxsize=max_buffer_chunks)
    create_task(background_stream())
    await audio_ready.wait()
    while (audio := await queue.get()) is not STREAM_FINISHED: # Modified from https://stackoverflow.com/a/63974948
        yield audio

def stream_bytes_to_array(byte_stream: Generator[Buffer, int, None],
                          dtype: DTypeLike) -> Generator[ndarray, int, None]:
    """Convenience generator function to automatically convert any `byte_stream` into a numpy compatible sample stream. 

    Args:
        byte_stream (Generator[Buffer, int, None]): Any Buffer generator.
        dtype (DTypeLike): The underlying audio sample format as a numpy dtype.

    Yields:
        Generator[ndarray, int, None]: Audio data formatted as a numpy array.
    """
    # TODO: This is a near copy of `stream_match_audio_channels()`. Consider making this less DRY?
    num_frames = yield b""
    while True:
        try:
            audio = byte_stream.send(num_frames) 
            num_frames = yield np.frombuffer(audio, dtype=dtype)  # If ArrayLike is passed here, then lossy compression occurs at worse-case.
        except StopIteration:
            break

def stream_sentinel() -> Generator[ArrayLike, int, None]:
    """Convenience generator function to simply yield nothing. Typically used agaisnt race conditions when track audio data is requested
    before the complete audio stream generator is initialized.

    Yields:
        Generator[ArrayLike, int, None]: Empty audio data
    """
    num_frames = yield b""
    while True:
        try:
            num_frames = yield np.zeros((num_frames, 1))
        except StopIteration:
            break

def stream_pad(ndarray_stream: Generator[ndarray, int, None], channels: int) -> Generator[ndarray, int, None]:
    """When calculating np.average to mix multiple audio streams,
    the function assumes all audio streams are identical in shape.
    This is the case until an audio stream is nearing its end, whereby
    it returns an trimmed audio stream. pad() ensures that the
    trimmed audio stream is padded.

    Args:
        sample_stream (Generator[ndarray, int, None]): Any synchronous audio generator whose data is formatted as a numpy array.
        channels (int): Number of audio channels.

    Yields:
        Generator[ndarray, int, None]:formatted as a numpy array.
    """
    num_frames = yield b""
    while True:
        try:
            audio = ndarray_stream.send(num_frames) 
            if not (audio.ndim and audio.size):
                num_frames = yield np.zeros((num_frames, channels))
            elif audio.shape[0] != num_frames:
                padded = audio.copy()
                padded.resize((num_frames, channels))
                num_frames = yield padded
            else:
                num_frames = yield audio
        except StopIteration:
            break

def stream_handle_mute(sample_stream: Generator[ArrayLike, int, None], track: Track) -> Generator[ArrayLike, int, None]:
    """Convenience generator function to purposely throw out audio data if `track` is muted, creating the effect of played but unheard audio.

    Args:
        sample_stream (Generator[ArrayLike, int, None]): Any synchronous audio generator
        track (Track): Any `Track` class.

    Yields:
        Generator[ArrayLike, int, None]: Audio data
    """
    num_frames = yield b""
    while True:
        try:
            audio = sample_stream.send(num_frames) 
            if track.muted:
                num_frames = yield np.zeros(np.shape(audio)) # NOTE: This is faster than `np.zeros_like(x)`, verify this by modifying the question `timeit` script and testing it agaisnt `np.zeroes(np.shape(audio))` from https://stackoverflow.com/questions/27464039/why-the-performance-difference-between-numpy-zeros-and-numpy-zeros-like
            else:
                num_frames = yield audio
        except StopIteration:
            break

In = TypeVar('In') 
Out = TypeVar('Out')
Params = ParamSpec('P')
AudioGenerator = Generator[Out, int, None]
GeneratorFactory: TypeAlias = Callable[Concatenate[In, Params], AudioGenerator]
Args: TypeAlias = Tuple[Any, ...]
Kwargs: TypeAlias = Dict[str, Any]
Transform: TypeAlias = Tuple[GeneratorFactory, Args, Kwargs]

class AudioPipeline: # NOTE: All of AudioPipeline has been AI-generated and tested + verified for correctness

    def __init__(self, *transforms: Transform):
        """
            Immutable pipeline that preserves order and chains audio processing transformations.

            Creates a composable pipeline of generator transformations that can be applied to any source.
            Each transformation in the pipeline receives the output of the previous transformation,
            creating a lazy evaluation chain that processes audio data efficiently. 

            Each generator must contain an initilization yield as each of them will be started in the pipeline.

            Args:
                *transforms: Variable number of (function, args, kwargs) tuples representing
                    transformations to apply. Each function should return an AudioGenerator with
                    an initialization yield. Typically not called directly - use >> operator instead.

            Usage:
                # Example audio generator with initialization and send capability
                def amplify(source, factor=2):
                    # Initialization
                    current_factor = factor
                    yield  # Initialization yield
                    
                    # Main processing loop
                    for sample in source:
                        # Can receive new amplification factor via send()
                        new_factor = yield sample * current_factor
                        if new_factor is not None:
                            current_factor = new_factor
                
                # Create a pipeline using the >> operator
                pipeline = (AudioPipeline()
                    >> (amplify, 2)
                    >> bandpass_filter
                    >> compressor)

                # Apply to audio stream
                audio_gen = pipeline(audio_samples)

                # Compile for reuse
                process = pipeline.compile()
                output = list(process(another_stream))
        """
        self.transforms = transforms

    def __rshift__(self, transform: Union[GeneratorFactory, Transform]) -> 'AudioPipeline':
        """
        Add a transformation to the pipeline using the >> operator.

        Creates a new Pipeline instance with the additional transformation appended,
        preserving immutability. The transformation should be a generator function
        that yields once for initialization and can accept integer values via send().

        Args:
            transform: Either a generator function or a tuple of (generator_function, *args, **kwargs).
                - If callable: Applied with no additional arguments
                - If tuple: First element is the generator function, remaining elements are arguments
                - If last tuple element is a dict, it's treated as kwargs

            Generator functions in the pipeline first positional argument must be either:
                - The initial `source` passed to the pipeline (for the first transform)
                - The output from the previous transform in the chain (for subsequent transforms)

        Returns:
            Pipeline: New Pipeline instance with the transformation added.

        Raises:
            TypeError: If transform is neither callable nor tuple.
        """
        if callable(transform):
            return AudioPipeline(*self.transforms, (transform, (), {}))
        elif isinstance(transform, tuple):
            func = transform[0]
            args = transform[1:] if len(transform) > 1 else ()
            kwargs = {}
            if args and isinstance(args[-1], dict):
                args, kwargs = args[:-1], args[-1]
            return AudioPipeline(*self.transforms, (func, args, kwargs))
        else:
            raise TypeError(f"Generator {transform} must be callable or tuple")

    def __call__(self, source: In) -> AudioGenerator:
        """
        Apply the pipeline to a source.

        Executes all transformations in order, passing the output of each
        transformation as input to the next. Each generator will be `GEN_STARTED` by having their first
        yield consumed before processing begins.

        Args:
            source: The input to process through the pipeline. Can be any type
                that the first generator in the chain accepts (iterable, generator, etc).

        Returns:
            AudioGenerator: Curried `AudioGenerator` of `transforms`

        """
        result = source
        for func, args, kwargs in self.transforms:
            result = func(result, *args, **kwargs)
            next(result)  # Consume initialization yield
        return result

    def compile(self) -> GeneratorFactory:
        """
        Compile the pipeline into a reusable generator factory.
        
        Returns a function that can be called multiple times with different
        sources. Each call will create a new chain of initialized generators
        that process the input and support send() communication.
        
        Returns:
            GeneratorFactory: Function that returns a curried `AudioGenerator` of `transforms`
        """
        return self.__call__

    def __repr__(self):
        """
        Return a string representation of the pipeline for debugging.

        Shows the sequence of transformations in a readable format using >> notation.
        Functions with arguments are displayed with ellipsis to indicate parameters.

        Returns:
            str: Human-readable representation of the pipeline structure.
        """
        cls_name = type(self).__name__
        if not self.transforms:
            return f"{cls_name}()"

        steps = []
        for func, args, kwargs in self.transforms:
            name = func.__name__ if hasattr(func, "__name__") else str(func)
            if args or kwargs:
                steps.append(f"{name}(...)")
            else:
                steps.append(name)

        return f"{cls_name}({' >> '.join(steps)})"