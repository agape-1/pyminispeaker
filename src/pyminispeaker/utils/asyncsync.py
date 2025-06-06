# Typing
from typing import Any, Literal, Coroutine, Optional, AsyncGenerator, Generator, Callable
from asyncio import AbstractEventLoop

# Helpers
from asyncio import get_event_loop, run_coroutine_threadsafe, get_running_loop, to_thread

# Main dependencies
from threading import Event as ThreadEvent
from collections import deque

def poll_async_generator(stream: AsyncGenerator[Any, Any], default_empty_factory: Callable[[], Any] = lambda : None, loop: Optional[AbstractEventLoop] = None) -> Generator[Any, None, None]:
    """
    Converts a asychronous generator to a synchronous one via polling. In other words, always return None when the asychronous generator is not ready.

    Args:
        stream (AsyncGenerator[Any, Any, Any]): Any AsyncGenerator.
        default_empty_factory (Callable[[], Any]): The value to return when the asychronous value is not ready. Defaults to returning None.
        loop (Optional[AbstractEventLoop]): The asyncio loop. Defaults to `get_event_loop()`.
    Yields:
        Generator[Any, Any, None]: A sychronous generator.
    """
    # TODO: Figure out how to make a `send() consumer retrieve from `asend()` synchronously for "The stream will `send()` a value whenever possible."
    if loop is None:
        loop = get_event_loop()
    buffer = deque() # Asked ChatGPT to help rename some of these variables
    async def stream_to_buffer():
        async for item in stream:
            buffer.append(item)
    collect_items = run_coroutine_threadsafe( 
        stream_to_buffer(), 
    loop) # We should not use call_soon_threadsafe/run_coroutine_threadsafe on `__anext__()`, theoretically giving a asychronous generator may allow lower level optimizations verseus manually calling __anext__() unpredictably?  
    while not collect_items.done() or buffer:
        if buffer:
            yield buffer.popleft() # Space complexity of deque is O(1) because a sychronous polling consumer will almost always consume faster than a asychronous producer can provide. 
        else:
            yield default_empty_factory()

class Event():
    """Class implementing event objects, that will dynamically determines which method(asynchronous/synchronous) to wait.

Events manage a flag that can be set to true with the set() method and reset
to false with the clear() method. The wait() method blocks until the flag is
true. The flag is initially false.

Warning:
This class will automatically determine what 
    """
    def __init__(self):
        """Initializes the underlying asycnhronous and thread event objects, publicly accessible as Event().tevent.
        """
        self.tevent = ThreadEvent()

    @property
    def _async(self) -> bool:
        """
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
        """Set the internal flag to true. All threads and coroutines waiting for it to
become true are awakened. Coroutines and Threads that call wait() once the flag is
true will not block at all.
        """
        self.tevent.set()

    def wait(self, timeout: Optional[float] = None) -> bool | Coroutine[Any, Any, Literal[True]]:
        """Dynamically determines which method(asynchronous/synchronous) to wait. 

        Args:
            timeout (Optional[float]): A
floating point number specifying a timeout for the operation in seconds
(or fractions thereof) to block. Defaults to None.

        Returns:
            bool | Coroutine[Any, Any, Literal[True]]: If no asychronous loop is present, wait identical to threading.Event().wait(). Otherwise, return an equivalent coroutine of Event().wait().
        """
        if self._async:
            return to_thread(self.tevent.wait, timeout=timeout)
        return self.tevent.wait(timeout=timeout)
    
    def is_set(self) -> bool:
        """Return True if and only if the internal flag is true.
        """
        return self.tevent.is_set()
