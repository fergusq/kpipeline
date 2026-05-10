import asyncio
from typing import Sequence
import pytest
import time

from kpipeline.async_pipeline import AsyncSemaphorePipe, AsyncPipe


# ─── Test fixtures / helper pipes ──────────────────────────────────────────────


class AsyncAddPipe(AsyncPipe[int, int, int]):
    """Simple async pipe that adds metadata to input."""
    async def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d + metadata for d in data]


class AsyncSlowAddPipe(AsyncPipe[int, int, int]):
    """Async pipe with artificial delay to test concurrency limiting."""
    def __init__(self, delay: float = 0.1):
        self.delay = delay

    async def apply(self, data: int, metadata: int) -> int:
        await asyncio.sleep(self.delay)
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        await asyncio.sleep(self.delay)
        return [d + metadata for d in data]


class AsyncConcurrencyTrackingPipe(AsyncPipe[int, int, int]):
    """
    Tracks the peak number of concurrent executions of apply()
    and records batch_apply call sizes.
    """
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.current_concurrent = 0
        self.peak_concurrent = 0
        self.apply_calls: list[int] = []
        self.batch_apply_calls: list[int] = []
        self._lock = asyncio.Lock()

    async def apply(self, data: int, metadata: int) -> int:
        async with self._lock:
            self.current_concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.current_concurrent)
            self.apply_calls.append(data)

        await asyncio.sleep(self.delay)

        async with self._lock:
            self.current_concurrent -= 1

        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        async with self._lock:
            self.batch_apply_calls.append(len(data))
            self.current_concurrent += len(data)
            self.peak_concurrent = max(self.peak_concurrent, self.current_concurrent)

        await asyncio.sleep(self.delay)

        async with self._lock:
            self.current_concurrent -= len(data)

        return [d + metadata for d in data]


class AsyncFailingPipe(AsyncPipe[int, int, int]):
    """Async pipe that always raises."""
    async def apply(self, data: int, metadata: int) -> int:
        raise ValueError(f"failed on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise ValueError(f"batch failed")


class AsyncFailOnNegativePipe(AsyncPipe[int, int, int]):
    """Fails on negative inputs."""
    async def apply(self, data: int, metadata: int) -> int:
        if data < 0:
            raise ValueError(f"negative: {data}")
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        for d in data:
            if d < 0:
                raise ValueError(f"negative in batch: {d}")
        return [d + metadata for d in data]


class AsyncBatchTrackingPipe(AsyncPipe[int, int, int]):
    """Tracks exactly which batches are passed to async_batch_apply."""
    def __init__(self):
        self.batches_received: list[list[int]] = []

    async def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        self.batches_received.append(list(data))
        return [d + metadata for d in data]


# ─── Basic apply() tests ──────────────────────────────────────────────────────


class TestAsyncSemaphorePipeApply:
    @pytest.mark.asyncio
    async def test_apply_returns_correct_result(self):
        """apply() passes through to subpipe and returns the correct result."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=5)
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_apply_single_item(self):
        """A single apply call works with any max_concurrent."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=1)
        result = await pipe.apply(7, 3)
        assert result == 10

    @pytest.mark.asyncio
    async def test_apply_propagates_exception(self):
        """Exceptions from the subpipe propagate through apply()."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailingPipe(), max_concurrent=5)
        with pytest.raises(ValueError, match="failed on 42"):
            await pipe.apply(42, 0)

    @pytest.mark.asyncio
    async def test_apply_releases_semaphore_on_success(self):
        """After a successful apply, the semaphore slot is released."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=1)

        # If semaphore wasn't released, the second call would hang
        async with asyncio.timeout(1.0):
            await pipe.apply(1, 0)
            await pipe.apply(2, 0)
            await pipe.apply(3, 0)

    @pytest.mark.asyncio
    async def test_apply_releases_semaphore_on_exception(self):
        """After a failed apply, the semaphore slot is still released."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailingPipe(), max_concurrent=1)

        async with asyncio.timeout(1.0):
            with pytest.raises(ValueError):
                await pipe.apply(1, 0)

            # Should not hang — semaphore was released despite exception
            with pytest.raises(ValueError):
                await pipe.apply(2, 0)


# ─── Concurrency limiting with apply() ────────────────────────────────────────


class TestAsyncSemaphorePipeConcurrencyApply:
    @pytest.mark.asyncio
    async def test_limits_concurrent_apply_calls(self):
        """Concurrent apply calls are limited to max_concurrent."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.05)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=3)

        async with asyncio.timeout(5.0):
            results = await asyncio.gather(*[pipe.apply(i, 0) for i in range(10)])

        assert tracker.peak_concurrent <= 3
        assert sorted(results) == list(range(10))

    @pytest.mark.asyncio
    async def test_max_concurrent_1_serializes_calls(self):
        """With max_concurrent=1, calls are effectively serialized."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.02)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=1)

        async with asyncio.timeout(5.0):
            results = await asyncio.gather(*[pipe.apply(i, 0) for i in range(5)])

        assert tracker.peak_concurrent == 1
        assert sorted(results) == list(range(5))

    @pytest.mark.asyncio
    async def test_concurrency_limit_timing(self):
        """
        With max_concurrent=2 and 4 tasks each taking 50ms,
        total time should be ~100ms (2 rounds), not 200ms (sequential) or 50ms (all parallel).
        """
        subpipe = AsyncSlowAddPipe(delay=0.05)
        pipe = AsyncSemaphorePipe(subpipe=subpipe, max_concurrent=2)

        start = time.monotonic()
        async with asyncio.timeout(2.0):
            results = await asyncio.gather(*[pipe.apply(i, 0) for i in range(4)])
        elapsed = time.monotonic() - start

        assert sorted(results) == [0, 1, 2, 3]
        # Should take ~100ms (2 rounds of 2), not 200ms (4 sequential)
        assert elapsed >= 0.09  # at least 2 rounds
        assert elapsed < 0.25  # but not fully sequential

    @pytest.mark.asyncio
    async def test_high_concurrency_limit_allows_all_parallel(self):
        """When max_concurrent >= number of tasks, all run in parallel."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.05)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=100)

        start = time.monotonic()
        async with asyncio.timeout(2.0):
            results = await asyncio.gather(*[pipe.apply(i, 0) for i in range(10)])
        elapsed = time.monotonic() - start

        assert tracker.peak_concurrent == 10
        assert elapsed < 0.15  # All parallel, ~50ms


# ─── async_batch_apply tests: data fits ────────────────────────────────────────


class TestAsyncSemaphorePipeBatchFits:
    @pytest.mark.asyncio
    async def test_batch_within_limit_returns_correct_results(self):
        """When batch size <= max_concurrent, results are correct."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=5)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_batch_equal_to_limit(self):
        """Batch exactly equal to max_concurrent works."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=3)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_batch_of_one(self):
        """A single-item batch works."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=5)
        results = await pipe.async_batch_apply([42], 1)
        assert list(results) == [43]

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """An empty batch returns empty results."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=5)
        results = await pipe.async_batch_apply([], 10)
        assert list(results) == []

    @pytest.mark.asyncio
    async def test_batch_acquires_semaphore_slots(self):
        """
        When batch_apply runs, it acquires semaphore slots so that
        concurrent apply() calls respect the limit.
        """
        tracker = AsyncConcurrencyTrackingPipe(delay=0.05)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=3)

        # Run a batch of 3 (fills all slots) concurrently with individual applies
        async with asyncio.timeout(3.0):
            batch_task = asyncio.create_task(pipe.async_batch_apply([1, 2, 3], 0))
            # Give batch time to acquire semaphore
            await asyncio.sleep(0.01)
            # These should wait until batch releases
            apply_task = asyncio.create_task(pipe.apply(99, 0))
            await asyncio.gather(batch_task, apply_task)

        # Peak should not exceed max_concurrent
        assert tracker.peak_concurrent <= 3

    @pytest.mark.asyncio
    async def test_batch_releases_semaphore_on_success(self):
        """After a successful batch, semaphore slots are released."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=2)

        async with asyncio.timeout(2.0):
            await pipe.async_batch_apply([1, 2], 0)
            # If slots weren't released, this would hang
            await pipe.async_batch_apply([3, 4], 0)

    @pytest.mark.asyncio
    async def test_batch_releases_semaphore_on_exception(self):
        """After a failed batch, semaphore slots are still released."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailingPipe(), max_concurrent=2)

        async with asyncio.timeout(2.0):
            with pytest.raises(ValueError):
                await pipe.async_batch_apply([1, 2], 0)

            # Should not hang — slots were released
            with pytest.raises(ValueError):
                await pipe.async_batch_apply([3, 4], 0)


# ─── async_batch_apply tests: data exceeds limit (chunking) ───────────────────


class TestAsyncSemaphorePipeBatchChunking:
    @pytest.mark.asyncio
    async def test_large_batch_returns_correct_results(self):
        """When batch > max_concurrent, results are still correct and in order."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=3)
        data = list(range(10))
        results = await pipe.async_batch_apply(data, 5)
        assert list(results) == [d + 5 for d in data]

    @pytest.mark.asyncio
    async def test_chunks_are_correct_size(self):
        """Large batches are split into chunks of max_concurrent size."""
        tracker = AsyncBatchTrackingPipe()
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=3)

        await pipe.async_batch_apply(list(range(7)), 0)

        # 7 items / 3 max = chunks of [3, 3, 1]
        assert tracker.batches_received == [[0, 1, 2], [3, 4, 5], [6]]

    @pytest.mark.asyncio
    async def test_chunks_exactly_divisible(self):
        """When batch is exactly divisible by max_concurrent, no remainder chunk."""
        tracker = AsyncBatchTrackingPipe()
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=3)

        await pipe.async_batch_apply(list(range(9)), 0)

        assert tracker.batches_received == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    @pytest.mark.asyncio
    async def test_chunks_processed_sequentially(self):
        """Chunks are processed one after another, not all at once."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.03)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=2)

        async with asyncio.timeout(3.0):
            results = await pipe.async_batch_apply(list(range(6)), 0)

        # 6 items in chunks of 2 → 3 sequential rounds
        assert tracker.peak_concurrent <= 2
        assert list(results) == list(range(6))

    @pytest.mark.asyncio
    async def test_chunking_timing(self):
        """
        With max_concurrent=2 and 6 items (3 chunks), each taking 50ms:
        total should be ~150ms (3 sequential rounds).
        """
        subpipe = AsyncSlowAddPipe(delay=0.05)
        pipe = AsyncSemaphorePipe(subpipe=subpipe, max_concurrent=2)

        start = time.monotonic()
        async with asyncio.timeout(3.0):
            results = await pipe.async_batch_apply(list(range(6)), 0)
        elapsed = time.monotonic() - start

        assert list(results) == list(range(6))
        # 3 chunks * 50ms each = 150ms minimum
        assert elapsed >= 0.13
        # But should not be much more
        assert elapsed < 0.35

    @pytest.mark.asyncio
    async def test_large_batch_order_preserved(self):
        """Order is preserved across chunk boundaries."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=4)
        data = list(range(100))
        results = await pipe.async_batch_apply(data, 1)
        assert list(results) == [d + 1 for d in data]

    @pytest.mark.asyncio
    async def test_chunked_batch_exception_in_first_chunk(self):
        """If the first chunk fails, the exception propagates."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailOnNegativePipe(), max_concurrent=2)
        with pytest.raises(ValueError, match="negative in batch"):
            await pipe.async_batch_apply([-1, 2, 3, 4], 0)

    @pytest.mark.asyncio
    async def test_chunked_batch_exception_in_later_chunk(self):
        """If a later chunk fails, the exception propagates."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailOnNegativePipe(), max_concurrent=2)
        # -5 is in the second chunk [3, -5]
        with pytest.raises(ValueError, match="negative in batch"):
            await pipe.async_batch_apply([1, 2, 3, -5], 0)

    @pytest.mark.asyncio
    async def test_chunked_batch_releases_semaphore_on_exception(self):
        """Even if a chunk fails, semaphore slots are released for future calls."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailOnNegativePipe(), max_concurrent=2)

        async with asyncio.timeout(2.0):
            with pytest.raises(ValueError):
                await pipe.async_batch_apply([-1, 2, 3, 4], 0)

            # Should still work after failure
            results = await pipe.async_batch_apply([5, 6], 0)
            assert list(results) == [5, 6]


# ─── Interaction between apply() and async_batch_apply() ──────────────────────


class TestAsyncSemaphorePipeInteraction:
    @pytest.mark.asyncio
    async def test_apply_and_batch_respect_shared_semaphore(self):
        """
        apply() and async_batch_apply() share the same semaphore.
        Concurrent usage should not exceed max_concurrent.
        """
        tracker = AsyncConcurrencyTrackingPipe(delay=0.03)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=3)

        async with asyncio.timeout(3.0):
            # Mix individual applies and batches
            results = await asyncio.gather(
                pipe.apply(1, 0),
                pipe.apply(2, 0),
                pipe.async_batch_apply([3, 4], 0),
                pipe.apply(5, 0),
            )

        assert tracker.peak_concurrent <= 3

    @pytest.mark.asyncio
    async def test_many_concurrent_applies_with_batch(self):
        """Stress test: many applies + batches all respecting the limit."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.02)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=4)

        async with asyncio.timeout(5.0):
            tasks = []
            for i in range(10):
                tasks.append(pipe.apply(i, 0))
            tasks.append(pipe.async_batch_apply(list(range(10, 14)), 0))
            tasks.append(pipe.async_batch_apply(list(range(14, 20)), 0))

            await asyncio.gather(*tasks)

        assert tracker.peak_concurrent <= 4


# ─── max_concurrent edge values ───────────────────────────────────────────────


class TestAsyncSemaphorePipeEdgeConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrent_1_apply(self):
        """With max_concurrent=1, only one apply runs at a time."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.02)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=1)

        async with asyncio.timeout(3.0):
            await asyncio.gather(*[pipe.apply(i, 0) for i in range(5)])

        assert tracker.peak_concurrent == 1

    @pytest.mark.asyncio
    async def test_max_concurrent_1_batch(self):
        """With max_concurrent=1, batch is split into single-item chunks."""
        tracker = AsyncBatchTrackingPipe()
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=1)

        await pipe.async_batch_apply([1, 2, 3], 0)

        assert tracker.batches_received == [[1], [2], [3]]

    @pytest.mark.asyncio
    async def test_large_max_concurrent(self):
        """A very large max_concurrent effectively allows unlimited concurrency."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.02)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=1000)

        async with asyncio.timeout(2.0):
            await asyncio.gather(*[pipe.apply(i, 0) for i in range(20)])

        assert tracker.peak_concurrent == 20


# ─── No deadlock tests ─────────────────────────────────────────────────────────


class TestAsyncSemaphorePipeNoDeadlock:
    @pytest.mark.asyncio
    async def test_no_deadlock_sequential_batches(self):
        """Sequential batch calls do not deadlock."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncSlowAddPipe(delay=0.02), max_concurrent=2)

        async with asyncio.timeout(3.0):
            for _ in range(5):
                results = await pipe.async_batch_apply([1, 2, 3], 0)
                assert list(results) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_no_deadlock_concurrent_batches(self):
        """Concurrent batch calls do not deadlock."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncSlowAddPipe(delay=0.02), max_concurrent=2)

        async with asyncio.timeout(5.0):
            results = await asyncio.gather(
                pipe.async_batch_apply([1, 2, 3], 0),
                pipe.async_batch_apply([4, 5, 6], 0),
                pipe.async_batch_apply([7, 8], 0),
            )

        assert list(results[0]) == [1, 2, 3]
        assert list(results[1]) == [4, 5, 6]
        assert list(results[2]) == [7, 8]

    @pytest.mark.asyncio
    async def test_no_deadlock_mixed_applies_and_batches(self):
        """Mixing apply() and async_batch_apply() does not deadlock."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncSlowAddPipe(delay=0.02), max_concurrent=2)

        async with asyncio.timeout(5.0):
            tasks = [
                pipe.apply(1, 0),
                pipe.async_batch_apply([2, 3, 4], 0),
                pipe.apply(5, 0),
                pipe.async_batch_apply([6, 7], 0),
                pipe.apply(8, 0),
            ]
            await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_no_deadlock_after_exception(self):
        """After exceptions, the pipe does not deadlock on subsequent calls."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailOnNegativePipe(), max_concurrent=2)

        async with asyncio.timeout(3.0):
            for _ in range(5):
                with pytest.raises(ValueError):
                    await pipe.apply(-1, 0)

                # Should still work
                result = await pipe.apply(1, 10)
                assert result == 11

    @pytest.mark.asyncio
    async def test_no_deadlock_batch_exception_then_apply(self):
        """After a batch exception, individual applies still work."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncFailOnNegativePipe(), max_concurrent=3)

        async with asyncio.timeout(3.0):
            with pytest.raises(ValueError):
                await pipe.async_batch_apply([1, -2, 3], 0)

            # Semaphore should be fully released — these should not deadlock
            results = await asyncio.gather(
                pipe.apply(10, 0),
                pipe.apply(20, 0),
                pipe.apply(30, 0),
            )
            assert sorted(results) == [10, 20, 30]


# ─── Graph / visualization tests ──────────────────────────────────────────────


class TestAsyncSemaphorePipeGraph:
    def test_to_graph_returns_valid_graph(self):
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=5)
        graph = pipe.to_graph()
        assert graph.is_valid()

    def test_is_wrapper(self):
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=5)
        assert pipe.is_wrapper() is True

    def test_get_subgraph_returns_subpipe_graph(self):
        subpipe = AsyncAddPipe()
        pipe = AsyncSemaphorePipe(subpipe=subpipe, max_concurrent=5)
        subgraph = pipe.get_subgraph()
        assert subgraph is not None
        assert subgraph.is_valid()

    def test_to_node_includes_max_concurrent(self):
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=7, description="Rate limit")
        node = pipe.to_node()
        assert "Rate limit" in node.title
        assert "max_concurrent=7" in node.title

    def test_to_dot_does_not_crash(self):
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=3)
        graph = pipe.to_graph()
        dot_str = graph.to_dot()
        assert isinstance(dot_str, str)
        assert len(dot_str) > 0


# ─── Composability tests ──────────────────────────────────────────────────────


class TestAsyncSemaphorePipeComposability:
    @pytest.mark.asyncio
    async def test_chain_with_another_pipe(self):
        """AsyncSemaphorePipe can be chained with other pipes."""
        semaphore_pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=3)

        class AsyncDoublePipe(AsyncPipe[int, int, int]):
            async def apply(self, data: int, metadata: int) -> int:
                return data * 2

        chain = semaphore_pipe | AsyncDoublePipe()
        result = await chain.apply(3, 10)
        # Add: 3+10=13, then double: 13*2=26
        assert result == 26

    @pytest.mark.asyncio
    async def test_nested_semaphore_pipes(self):
        """A semaphore pipe can wrap another semaphore pipe."""
        inner = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=5)
        outer = AsyncSemaphorePipe(subpipe=inner, max_concurrent=2)

        async with asyncio.timeout(2.0):
            results = await asyncio.gather(*[outer.apply(i, 1) for i in range(6)])

        assert sorted(results) == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_wrapping_slow_pipe(self):
        """Wrapping a slow pipe correctly limits throughput."""
        tracker = AsyncConcurrencyTrackingPipe(delay=0.03)
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=2)

        async with asyncio.timeout(3.0):
            await asyncio.gather(*[pipe.apply(i, 0) for i in range(8)])

        assert tracker.peak_concurrent <= 2


# ─── Edge cases ────────────────────────────────────────────────────────────────


class TestAsyncSemaphorePipeEdgeCases:
    @pytest.mark.asyncio
    async def test_batch_size_one_less_than_limit(self):
        """Batch of (max_concurrent - 1) uses fast path."""
        tracker = AsyncBatchTrackingPipe()
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=5)

        await pipe.async_batch_apply([1, 2, 3, 4], 0)
        # Should be a single batch call (no chunking)
        assert tracker.batches_received == [[1, 2, 3, 4]]

    @pytest.mark.asyncio
    async def test_batch_size_one_more_than_limit(self):
        """Batch of (max_concurrent + 1) triggers chunking."""
        tracker = AsyncBatchTrackingPipe()
        pipe = AsyncSemaphorePipe(subpipe=tracker, max_concurrent=3)

        await pipe.async_batch_apply([1, 2, 3, 4], 0)
        # Chunked: [1,2,3] then [4]
        assert tracker.batches_received == [[1, 2, 3], [4]]

    @pytest.mark.asyncio
    async def test_metadata_passed_correctly(self):
        """Metadata is correctly passed through to the subpipe in all paths."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=2)

        # Small batch (fast path)
        results = await pipe.async_batch_apply([1, 2], 100)
        assert list(results) == [101, 102]

        # Large batch (chunked)
        results = await pipe.async_batch_apply([1, 2, 3], 200)
        assert list(results) == [201, 202, 203]

    @pytest.mark.asyncio
    async def test_large_batch_stress(self):
        """Stress test with a large number of items."""
        pipe = AsyncSemaphorePipe(subpipe=AsyncAddPipe(), max_concurrent=10)
        data = list(range(500))
        results = await pipe.async_batch_apply(data, 1)
        assert list(results) == [d + 1 for d in data]

    @pytest.mark.asyncio
    async def test_custom_description(self):
        """Custom description is stored correctly."""
        pipe = AsyncSemaphorePipe(
            subpipe=AsyncAddPipe(),
            max_concurrent=5,
            description="My rate limiter",
        )
        assert pipe.description == "My rate limiter"
        node = pipe.to_node()
        assert "My rate limiter" in node.title
