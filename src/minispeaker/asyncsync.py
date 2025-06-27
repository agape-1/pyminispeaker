# Typing
from __future__ import annotations
from typing_extensions import TypeVar, Any, Literal, Coroutine, Optional, AsyncGenerator, Generator, Callable
from asyncio import AbstractEventLoop

# Helpers
from asyncio import get_event_loop, run_coroutine_threadsafe, get_running_loop

# Main dependencies
from threading import Event as ThreadEvent
from collections import deque


async def _to_thread(func, /, *args, **kwargs):  # Python 3.8 does not have `to_thread`: Copied from https://github.com/python/cpython/blob/main/Lib/asyncio/threads.py#L12
    """Asynchronously run function *func* in a separate thread.

    Any *args and **kwargs supplied for this function are directly passed
    to *func*. Also, the current :class:`contextvars.Context` is propagated,
    allowing context variables from the main thread to be accessed in the
    separate thread.

    Return a coroutine that can be awaited to get the eventual result of *func*.
    """
    import asyncio
    import contextvars
    import functools
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)

T = TypeVar('T')
S = TypeVar('S')
E = TypeVar('E')


def poll_async_generator(stream: AsyncGenerator[T, S], default_empty_factory: Callable[[], E] = lambda: None, loop: Optional[AbstractEventLoop] = None) -> Generator[T | E, S, None]:
    """Converts a asychronous generator `stream` into a synchronous one via polling.

    Args:
        stream (AsyncGenerator[T, S]): Any `Asyncgenerator`.
        default_empty_factory (Callable[[], E], optional): A function whose return value is used when the next polled data from `stream` is not available. Defaults to lambda:None.
        loop (Optional[AbstractEventLoop], optional): The event loop containing `stream`. Defaults to `get_event_loop()``

    Returns:
        Generator[T | E, S, None]: A `stream` equivalent synchronous generator

    Yields:
        T | E: When no stream data is available, return `E`. Otherwise fetch `T`.
    """
    # TODO: Figure out how to make a `send() consumer retrieve from `asend()` synchronously for "The stream will `send()` a value whenever possible."
    if loop is None:
        loop = get_event_loop()
    buffer = deque()  # Asked ChatGPT to help rename some of these variables

    async def stream_to_buffer():
        async for item in stream:
            buffer.append(item)
    collect_items = run_coroutine_threadsafe(
        stream_to_buffer(), loop)  # We should not use call_soon_threadsafe/run_coroutine_threadsafe on `__anext__()`, theoretically giving a asychronous generator may allow lower level optimizations verseus manually calling __anext__() unpredictably?
    while not collect_items.done() or buffer:
        if buffer:
            yield buffer.popleft()  # Space complexity of deque is O(1) because a sychronous polling consumer will almost always consume faster than a asychronous producer can provide.
        else:
            yield default_empty_factory()


class Event():
    """Class implementing event objects, that will dynamically determines which method(asynchronous/synchronous) to wait.

    Events manage a flag that can be set to true with the set() method and reset
    to false with the clear() method. The wait() method blocks until the flag is
    true. The flag is initially false.

    Warning:
    This class will automatically determine when to call a asychronous `wait` or synchronous `wait`.
    To directly use the sychronous `wait`, use `Event.tevent` like `Threading.Event`
    """
    def __init__(self):
        self.tevent = ThreadEvent()

    @property
    def _async(self) -> bool:
        """Attempts to detect if the `Event` is currently in a asynchronous loop.

        Returns:
            bool: Should the event be handled asynchronously?
        """
        try:
            return bool(get_running_loop())
        except RuntimeError:
            return False

    def clear(self):
        """Reset the internal flag to false.

        Subsequently, coroutines and threads calling wait() will block until set() is called to
        set the internal flag to true again.
        """
        self.tevent.clear()

    def set(self):
        """Set the internal flag to true. All threads and coroutines waiting for it to become true are awakened.

        Coroutines and Threads that call wait() once the flag is true will not block at all.
        """
        self.tevent.set()

    def wait(self, timeout: Optional[float] = None) -> bool | Coroutine[Any, Any, Literal[True]]:
        """Dynamically determines which method(asynchronous/synchronous) to wait.

        Args:
            timeout (Optional[float]): A floating point number specifying a timeout for the operation in seconds (or fractions thereof) to block. Defaults to None.

        Returns:
            bool | Coroutine[Any, Any, Literal[True]]: If no asychronous loop is present, wait identical to threading.Event().wait(). Otherwise, return an equivalent coroutine of Event().wait().
        """
        if self._async:
            return _to_thread(self.tevent.wait, timeout=timeout)
        return self.tevent.wait(timeout=timeout)

    def is_set(self) -> bool:
        """Return True if and only if the internal flag is true."""
        return self.tevent.is_set()
