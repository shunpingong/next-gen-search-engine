from __future__ import annotations

import asyncio
import itertools

from agent.models import FrontierTask


class FrontierScheduler:
    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[float, int, FrontierTask]] = asyncio.PriorityQueue()
        self._sequence = itertools.count()

    def push(self, task: FrontierTask) -> None:
        self._queue.put_nowait((-task.priority, next(self._sequence), task))

    async def pop_batch(self, *, limit: int, kind: str | None = None) -> list[FrontierTask]:
        if limit <= 0:
            return []

        tasks: list[FrontierTask] = []
        deferred: list[tuple[float, int, FrontierTask]] = []
        while len(tasks) < limit and not self._queue.empty():
            item = self._queue.get_nowait()
            _, _, task = item
            if kind and task.kind != kind:
                deferred.append(item)
                continue
            tasks.append(task)

        for item in deferred:
            self._queue.put_nowait(item)
        return tasks

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()
