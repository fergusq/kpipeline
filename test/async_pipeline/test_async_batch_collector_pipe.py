import asyncio
from typing import Sequence
import pytest

from kpipeline.async_pipeline import AsyncBatchCollectorPipe, AsyncPipe
from kpipeline.pipeline import Pipe


class BatchSumPipe(Pipe[int, int, int]):
    def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        total = sum(data) + metadata
        return [total] * len(data)


class AsyncBatchSumPipe(AsyncPipe[int, int, int]):
    async def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        await asyncio.sleep(0.01)
        total = sum(data) + metadata
        return [total] * len(data)


@pytest.mark.asyncio
async def test_async_batch_collector_pipe_sync_subpipe():
    subpipe = BatchSumPipe()
    collector: AsyncBatchCollectorPipe[int, int, int] = AsyncBatchCollectorPipe(subpipe, 0.1)
    
    # Use a task to ensure the timer runs in the background
    # Since AsyncBatchCollectorPipe starts the timer in __init__, it's already running.
    
    # We need to wait for the results.
    # Process multiple inputs with same metadata
    meta = 10
    results = await asyncio.gather(
        collector.apply(1, meta),
        collector.apply(2, meta),
        collector.apply(3, meta),
    )
    
    # Expected: sum(1, 2, 3) + 10 = 16
    assert results == [16, 16, 16]


@pytest.mark.asyncio
async def test_async_batch_collector_pipe_async_subpipe():
    subpipe = AsyncBatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.1)
    
    meta = 5
    results = await asyncio.gather(
        collector.apply(1, meta),
        collector.apply(2, meta),
    )
    
    # Expected: sum(1, 2) + 5 = 8
    assert results == [8, 8]


@pytest.mark.asyncio
async def test_async_batch_collector_pipe_different_metadata():
    subpipe = BatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.1)
    
    meta1 = 10
    meta2 = 20
    
    results = await asyncio.gather(
        collector.apply(1, meta1),
        collector.apply(2, meta1),
        collector.apply(1, meta2),
        collector.apply(2, meta2),
    )
    
    # meta1: 1+2+10 = 13
    # meta2: 1+2+20 = 23
    assert results == [13, 13, 23, 23]


@pytest.mark.asyncio
async def test_async_batch_collector_pipe_timing():
    subpipe = BatchSumPipe()
    # Short timeout
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)
    
    meta = 0
    
    # First batch
    r1 = await collector.apply(1, meta)
    await asyncio.sleep(0.1) # Wait for timer to fire and clear batch
    
    # Second batch
    r2 = await collector.apply(2, meta)
    
    # r1 should be just [1] since it was processed alone (or the timer fired before 2 arrived)
    # Wait, if we await apply(1, meta), it will wait until the timer fires.
    # If we sleep 0.1 after that, the next apply(2, meta) will be in a new batch.
    
    assert r1 == 1
    assert r2 == 2
