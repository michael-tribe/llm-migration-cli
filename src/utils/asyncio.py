import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.settings import settings


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


def run_async_tasks(tasks: list[Coroutine[Any, Any, Any]]) -> Any:
    return asyncio.run(gather_tasks(tasks))


async def gather_tasks(tasks: list[Coroutine[Any, Any, Any]]) -> Any:
    if settings.llm.max_concurrency == 1:
        return [await task for task in tasks]

    results = []
    for i in range(0, len(tasks), settings.llm.max_concurrency):
        results.extend(await asyncio.gather(*tasks[i : i + settings.llm.max_concurrency]))

    return results
