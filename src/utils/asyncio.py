import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


_executor = ThreadPoolExecutor(max_workers=1)


def run_sync(async_fn: Callable, *args: object, **kwargs: object) -> object:
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        future = _executor.submit(asyncio.run, async_fn(*args, **kwargs))
        return future.result()

    return asyncio.run(async_fn(*args, **kwargs))
