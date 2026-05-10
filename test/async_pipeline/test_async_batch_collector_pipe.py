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


class CountingPipe(Pipe[int, int, int]):
    """Tracks how many times batch_apply is called and with what sizes."""

    def __init__(self):
        self.batch_calls: list[int] = []

    def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        self.batch_calls.append(len(data))
        total = sum(data) + metadata
        return [total] * len(data)


class SlowPipe(AsyncPipe[int, int, int]):
    """A pipe whose batch_apply takes significant time, to test concurrent behavior."""

    async def apply(self, data: int, metadata: int) -> int:
        await asyncio.sleep(0.1)
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        await asyncio.sleep(0.1)
        total = sum(data) + metadata
        return [total] * len(data)


class FailingPipe(Pipe[int, int, int]):
    """A pipe that always raises an exception in batch_apply."""

    def apply(self, data: int, metadata: int) -> int:
        raise ValueError("intentional failure")

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise ValueError("intentional batch failure")


# ─── Deadlock detection ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deadlock_single_apply():
    """
    The original implementation deadlocks because apply() awaits a future
    while holding the lock, and the timer needs the lock to resolve the future.
    (This is actually not true, the deadlock never happened despite what the LLM says here.)
    This test catches it via a timeout.
    """
    subpipe = BatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(1.0):
        result = await collector.apply(42, 0)

    assert result == 42


@pytest.mark.asyncio
async def test_deadlock_concurrent_applies():
    """
    Multiple concurrent apply() calls should all resolve without deadlocking.
    With the buggy implementation, the first call holds the lock forever.
    """
    subpipe = BatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(2.0):
        results = await asyncio.gather(
            collector.apply(1, 10),
            collector.apply(2, 10),
            collector.apply(3, 10),
        )

    # sum(1,2,3) + 10 = 16
    assert results == [16, 16, 16]


@pytest.mark.asyncio
async def test_deadlock_sequential_applies():
    """
    Sequential calls must each resolve. If apply() holds the lock until the
    future resolves, the second call will hang waiting for the lock.
    """
    subpipe = BatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(2.0):
        r1 = await collector.apply(1, 0)
        r2 = await collector.apply(2, 0)

    assert r1 == 1
    assert r2 == 2


@pytest.mark.asyncio
async def test_deadlock_with_slow_subpipe():
    """
    Even when the subpipe takes time to process, callers should not deadlock.
    """
    subpipe = SlowPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(3.0):
        results = await asyncio.gather(
            collector.apply(1, 5),
            collector.apply(2, 5),
        )

    # sum(1,2) + 5 = 8
    assert results == [8, 8]


@pytest.mark.asyncio
async def test_deadlock_rapid_fire():
    """
    Rapidly submitting many items should not cause deadlock or lost items.
    """
    subpipe = BatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    n = 50
    meta = 0

    async with asyncio.timeout(5.0):
        results = await asyncio.gather(
            *[collector.apply(i, meta) for i in range(n)]
        )

    # All items in one batch: sum(0..49) + 0 = 1225
    expected_total = sum(range(n)) + meta
    assert all(r == expected_total for r in results)


# ─── Batching correctness ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_collector_batches_concurrent_items():
    """Items submitted concurrently within the time window are batched together."""
    subpipe = CountingPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.1)

    async with asyncio.timeout(2.0):
        results = await asyncio.gather(
            collector.apply(1, 10),
            collector.apply(2, 10),
            collector.apply(3, 10),
        )

    # All 3 should be in a single batch
    assert 3 in subpipe.batch_calls
    assert results == [16, 16, 16]


@pytest.mark.asyncio
async def test_batch_collector_separate_batches_by_time():
    """Items submitted in different time windows go into separate batches."""
    subpipe = CountingPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(3.0):
        # First batch
        r1 = await collector.apply(1, 0)
        # Wait for the timer to definitely fire and reset
        await asyncio.sleep(0.15)
        # Second batch
        r2 = await collector.apply(2, 0)

    assert r1 == 1  # batch of [1], sum=1
    assert r2 == 2  # batch of [2], sum=2
    # Should have been two separate batch_apply calls
    assert len(subpipe.batch_calls) == 2
    assert subpipe.batch_calls == [1, 1]


@pytest.mark.asyncio
async def test_batch_collector_different_metadata_groups():
    """Items with different metadata are processed in separate batch_apply calls."""
    subpipe = CountingPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.1)

    async with asyncio.timeout(2.0):
        results = await asyncio.gather(
            collector.apply(1, 10),
            collector.apply(2, 10),
            collector.apply(1, 20),
            collector.apply(2, 20),
        )

    # meta=10: sum(1,2)+10 = 13
    # meta=20: sum(1,2)+20 = 23
    assert results == [13, 13, 23, 23]
    # Two separate batch_apply calls (one per metadata group)
    assert sorted(subpipe.batch_calls) == [2, 2]


@pytest.mark.asyncio
async def test_batch_collector_async_subpipe():
    """Works correctly with an async subpipe."""
    subpipe = AsyncBatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.1)

    async with asyncio.timeout(2.0):
        results = await asyncio.gather(
            collector.apply(1, 5),
            collector.apply(2, 5),
        )

    assert results == [8, 8]


# ─── Error handling ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_collector_propagates_errors():
    """If the subpipe raises, all waiting futures should receive the exception."""
    subpipe = FailingPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(2.0):
        with pytest.raises(ValueError, match="intentional batch failure"):
            await asyncio.gather(
                collector.apply(1, 0),
                collector.apply(2, 0),
            )


@pytest.mark.asyncio
async def test_batch_collector_error_does_not_break_subsequent_batches():
    """
    After an error in one batch, subsequent batches should still work
    (the collector should not be left in a broken state).
    """
    call_count = 0

    class FailOncePipe(Pipe[int, int, int]):
        def apply(self, data: int, metadata: int) -> int:
            return data

        def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first batch fails")
            return [d + metadata for d in data]

    subpipe = FailOncePipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(3.0):
        # First batch fails
        with pytest.raises(ValueError, match="first batch fails"):
            await collector.apply(1, 0)

        # Wait for timer to reset
        await asyncio.sleep(0.15)

        # Second batch should succeed
        result = await collector.apply(5, 10)

    assert result == 15


# ─── Lifecycle management ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_collector_close_flushes_remaining():
    """Closing the collector should process any remaining accumulated items."""
    subpipe = CountingPipe()
    # Long timer — items would not be processed without explicit close
    collector = AsyncBatchCollectorPipe(subpipe, 10.0)

    async with asyncio.timeout(2.0):
        # Submit items without waiting for timer
        task = asyncio.create_task(collector.apply(42, 0))
        # Give the task time to register
        await asyncio.sleep(0.01)
        # Close should flush
        await collector.close()
        result = await task

    assert result == 42
    assert subpipe.batch_calls == [1]


@pytest.mark.asyncio
async def test_batch_collector_context_manager():
    """The collector works as an async context manager."""
    subpipe = BatchSumPipe()

    async with asyncio.timeout(2.0):
        async with AsyncBatchCollectorPipe(subpipe, 0.05) as collector:
            results = await asyncio.gather(
                collector.apply(1, 0),
                collector.apply(2, 0),
            )

    assert results == [3, 3]


@pytest.mark.asyncio
async def test_batch_collector_apply_after_close_raises():
    """Calling apply() after close() should raise an error."""
    subpipe = BatchSumPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 0.05)

    async with asyncio.timeout(2.0):
        await collector.close()

        with pytest.raises(RuntimeError, match="closed"):
            await collector.apply(1, 0)


# ─── Max batch size ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_collector_max_batch_size_triggers_early_flush():
    """When max_batch_size is reached, flush happens before the timer fires."""
    subpipe = CountingPipe()
    # Very long timer — would timeout if we relied on it
    collector = AsyncBatchCollectorPipe(subpipe, 10.0, max_batch_size=3)

    async with asyncio.timeout(2.0):
        results = await asyncio.gather(
            collector.apply(1, 0),
            collector.apply(2, 0),
            collector.apply(3, 0),
        )

    # Should have flushed immediately when batch size hit 3
    assert results == [6, 6, 6]
    assert subpipe.batch_calls == [3]

    await collector.close()


@pytest.mark.asyncio
async def test_batch_collector_max_batch_size_multiple_flushes():
    """Exceeding max_batch_size causes multiple flushes."""
    subpipe = CountingPipe()
    collector = AsyncBatchCollectorPipe(subpipe, 10.0, max_batch_size=2)

    async with asyncio.timeout(2.0):
        # Submit 4 items — should trigger 2 flushes of size 2
        # Note: timing means the first 2 might flush before the next 2 arrive
        r1, r2 = await asyncio.gather(
            collector.apply(1, 0),
            collector.apply(2, 0),
        )
        r3, r4 = await asyncio.gather(
            collector.apply(3, 0),
            collector.apply(4, 0),
        )

    assert r1 == 3  # sum(1,2)
    assert r2 == 3
    assert r3 == 7  # sum(3,4)
    assert r4 == 7
    assert subpipe.batch_calls == [2, 2]

    await collector.close()