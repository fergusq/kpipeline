import asyncio
from typing import Sequence
import pytest
import time

from kpipeline.async_pipeline import AsyncTimeoutPipe, AsyncPipe


# ─── Test fixtures / helper pipes ──────────────────────────────────────────────


class AsyncAddPipe(AsyncPipe[int, int, int]):
    """Simple async pipe that adds metadata to input."""
    async def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d + metadata for d in data]


class AsyncSlowPipe(AsyncPipe[int, int, int]):
    """Async pipe with configurable delay."""
    def __init__(self, delay: float = 1.0):
        self.delay = delay

    async def apply(self, data: int, metadata: int) -> int:
        await asyncio.sleep(self.delay)
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        await asyncio.sleep(self.delay)
        return [d + metadata for d in data]


class AsyncVariableDelayPipe(AsyncPipe[int, int, int]):
    """
    Async pipe whose delay depends on the input value.
    Delay = data * delay_factor seconds.
    """
    def __init__(self, delay_factor: float = 0.1):
        self.delay_factor = delay_factor

    async def apply(self, data: int, metadata: int) -> int:
        await asyncio.sleep(data * self.delay_factor)
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        # Batch takes as long as the slowest item
        max_delay = max(data) * self.delay_factor if data else 0
        await asyncio.sleep(max_delay)
        return [d + metadata for d in data]


class AsyncTrackingPipe(AsyncPipe[int, int, int]):
    """Tracks all calls and their completion status."""
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.apply_started: list[int] = []
        self.apply_completed: list[int] = []
        self.batch_started: list[list[int]] = []
        self.batch_completed: list[list[int]] = []

    async def apply(self, data: int, metadata: int) -> int:
        self.apply_started.append(data)
        await asyncio.sleep(self.delay)
        self.apply_completed.append(data)
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        self.batch_started.append(list(data))
        await asyncio.sleep(self.delay)
        self.batch_completed.append(list(data))
        return [d + metadata for d in data]


class AsyncFailingPipe(AsyncPipe[int, int, int]):
    """Async pipe that always raises ValueError."""
    async def apply(self, data: int, metadata: int) -> int:
        raise ValueError(f"failed on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise ValueError("batch failed")


class AsyncDelayThenFailPipe(AsyncPipe[int, int, int]):
    """Async pipe that waits then fails."""
    def __init__(self, delay: float = 0.5):
        self.delay = delay

    async def apply(self, data: int, metadata: int) -> int:
        await asyncio.sleep(self.delay)
        raise ValueError(f"delayed failure on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        await asyncio.sleep(self.delay)
        raise ValueError("delayed batch failure")


class AsyncCancellationTrackingPipe(AsyncPipe[int, int, int]):
    """Tracks whether the task was cancelled."""
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.was_cancelled: list[int] = []
        self.completed: list[int] = []

    async def apply(self, data: int, metadata: int) -> int:
        try:
            await asyncio.sleep(self.delay)
            self.completed.append(data)
            return data + metadata
        except asyncio.CancelledError:
            self.was_cancelled.append(data)
            raise

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        try:
            await asyncio.sleep(self.delay)
            self.completed.extend(data)
            return [d + metadata for d in data]
        except asyncio.CancelledError:
            self.was_cancelled.extend(data)
            raise


# ─── Basic apply() tests ──────────────────────────────────────────────────────


class TestAsyncTimeoutPipeApply:
    @pytest.mark.asyncio
    async def test_fast_subpipe_returns_result(self):
        """When the subpipe completes within the timeout, result is returned."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_slow_subpipe_raises_timeout(self):
        """When the subpipe exceeds the timeout, TimeoutError is raised."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.05)
        with pytest.raises(TimeoutError):
            await pipe.apply(3, 10)

    @pytest.mark.asyncio
    async def test_subpipe_just_within_timeout(self):
        """A subpipe that finishes just before the timeout succeeds."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=0.02), timeout=0.2)
        result = await pipe.apply(5, 10)
        assert result == 15

    @pytest.mark.asyncio
    async def test_timeout_does_not_delay_fast_operations(self):
        """Fast operations return immediately, not after the timeout duration."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=10.0)

        start = time.monotonic()
        await pipe.apply(1, 0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_subpipe_exception_propagates(self):
        """Non-timeout exceptions from the subpipe propagate normally."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncFailingPipe(), timeout=1.0)
        with pytest.raises(ValueError, match="failed on 42"):
            await pipe.apply(42, 0)

    @pytest.mark.asyncio
    async def test_exception_before_timeout_propagates(self):
        """If subpipe fails before timeout, the exception (not TimeoutError) is raised."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncDelayThenFailPipe(delay=0.02), timeout=1.0)
        with pytest.raises(ValueError, match="delayed failure on"):
            await pipe.apply(5, 0)

    @pytest.mark.asyncio
    async def test_timeout_beats_delayed_exception(self):
        """If timeout fires before the subpipe's own exception, TimeoutError is raised."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncDelayThenFailPipe(delay=1.0), timeout=0.05)
        with pytest.raises(TimeoutError):
            await pipe.apply(5, 0)


# ─── Cancellation behavior ────────────────────────────────────────────────────


class TestAsyncTimeoutPipeCancellation:
    @pytest.mark.asyncio
    async def test_timeout_cancels_subpipe(self):
        """When timeout fires, the subpipe's task is cancelled."""
        tracker = AsyncCancellationTrackingPipe(delay=1.0)
        pipe = AsyncTimeoutPipe(subpipe=tracker, timeout=0.05)

        with pytest.raises(TimeoutError):
            await pipe.apply(42, 0)

        # Give a moment for cancellation to propagate
        await asyncio.sleep(0.01)
        assert 42 not in tracker.completed

    @pytest.mark.asyncio
    async def test_timeout_does_not_cancel_fast_subpipe(self):
        """A subpipe that completes in time is not cancelled."""
        tracker = AsyncCancellationTrackingPipe(delay=0.01)
        pipe = AsyncTimeoutPipe(subpipe=tracker, timeout=1.0)

        result = await pipe.apply(7, 10)
        assert result == 17
        assert 7 in tracker.completed
        assert 7 not in tracker.was_cancelled

    @pytest.mark.asyncio
    async def test_elapsed_time_on_timeout(self):
        """The actual elapsed time should be approximately the timeout value."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=10.0), timeout=0.1)

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await pipe.apply(1, 0)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.08  # At least close to timeout
        assert elapsed < 0.3   # But not much longer


# ─── async_batch_apply with enable_batching=True ───────────────────────────────


class TestAsyncTimeoutPipeBatchingEnabled:
    @pytest.mark.asyncio
    async def test_fast_batch_returns_results(self):
        """When the batch completes within timeout, results are returned."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0, enable_batching=True)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_slow_batch_raises_timeout(self):
        """When the batch exceeds timeout, TimeoutError is raised."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.05, enable_batching=True)
        with pytest.raises(TimeoutError):
            await pipe.async_batch_apply([1, 2, 3], 10)

    @pytest.mark.asyncio
    async def test_single_timeout_cancels_entire_batch(self):
        """With batching enabled, timeout cancels the whole batch at once."""
        tracker = AsyncCancellationTrackingPipe(delay=1.0)
        pipe = AsyncTimeoutPipe(subpipe=tracker, timeout=0.05, enable_batching=True)

        with pytest.raises(TimeoutError):
            await pipe.async_batch_apply([1, 2, 3], 0)

        await asyncio.sleep(0.01)
        # None of the items should have completed
        assert tracker.completed == []

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """An empty batch returns empty results."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0, enable_batching=True)
        results = await pipe.async_batch_apply([], 10)
        assert list(results) == []

    @pytest.mark.asyncio
    async def test_batch_exception_propagates(self):
        """Non-timeout exceptions from batch_apply propagate."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncFailingPipe(), timeout=1.0, enable_batching=True)
        with pytest.raises(ValueError, match="batch failed"):
            await pipe.async_batch_apply([1, 2], 0)

    @pytest.mark.asyncio
    async def test_batch_timing(self):
        """With batching enabled, timeout applies to the entire batch processing time."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=0.05), timeout=0.2, enable_batching=True)
        # Batch takes 50ms total (one async_batch_apply call), well within 200ms
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]


# ─── async_batch_apply with enable_batching=False ──────────────────────────────


class TestAsyncTimeoutPipeBatchingDisabled:
    @pytest.mark.asyncio
    async def test_all_items_fast_returns_results(self):
        """When all items complete within timeout, results are returned."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0, enable_batching=False)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_all_items_slow_all_timeout(self):
        """When all items exceed timeout, all raise TimeoutError."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.05, enable_batching=False)
        with pytest.raises(TimeoutError):
            await pipe.async_batch_apply([1, 2, 3], 10)

    @pytest.mark.asyncio
    async def test_individual_timeout_per_item(self):
        """Each item has its own timeout — slow items fail, fast items succeed."""
        # Items with value 1 take 0.1s, items with value 5 take 0.5s
        pipe = AsyncTimeoutPipe(
            subpipe=AsyncVariableDelayPipe(delay_factor=0.1),
            timeout=0.2,
            enable_batching=False,
        )

        # Item 1: 0.1s (within 0.2s timeout) → succeeds
        # Item 5: 0.5s (exceeds 0.2s timeout) → TimeoutError
        # Since asyncio.gather raises on first exception by default,
        # the whole gather will raise
        with pytest.raises(TimeoutError):
            await pipe.async_batch_apply([1, 5], 10)

    @pytest.mark.asyncio
    async def test_items_run_concurrently(self):
        """With batching disabled, items run concurrently (not sequentially)."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=0.05), timeout=1.0, enable_batching=False)

        start = time.monotonic()
        results = await pipe.async_batch_apply([1, 2, 3, 4, 5], 0)
        elapsed = time.monotonic() - start

        assert list(results) == [1, 2, 3, 4, 5]
        # If concurrent: ~50ms. If sequential: ~250ms
        assert elapsed < 0.2

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """An empty batch returns empty results."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0, enable_batching=False)
        results = await pipe.async_batch_apply([], 10)
        assert list(results) == []

    @pytest.mark.asyncio
    async def test_single_item_batch(self):
        """A single-item batch works correctly."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0, enable_batching=False)
        results = await pipe.async_batch_apply([42], 1)
        assert list(results) == [43]

    @pytest.mark.asyncio
    async def test_single_item_timeout(self):
        """A single-item batch that times out raises correctly."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.05, enable_batching=False)
        with pytest.raises(TimeoutError):
            await pipe.async_batch_apply([42], 1)


# ─── Comparison: batching enabled vs disabled ──────────────────────────────────


class TestAsyncTimeoutPipeBatchingComparison:
    @pytest.mark.asyncio
    async def test_batching_enabled_uses_batch_apply(self):
        """With batching enabled, subpipe.async_batch_apply is called."""
        tracker = AsyncTrackingPipe(delay=0.0)
        pipe = AsyncTimeoutPipe(subpipe=tracker, timeout=1.0, enable_batching=True)

        await pipe.async_batch_apply([1, 2, 3], 0)

        assert tracker.batch_started == [[1, 2, 3]]
        assert tracker.apply_started == []

    @pytest.mark.asyncio
    async def test_batching_disabled_uses_individual_apply(self):
        """With batching disabled, individual apply() is called per item."""
        tracker = AsyncTrackingPipe(delay=0.0)
        pipe = AsyncTimeoutPipe(subpipe=tracker, timeout=1.0, enable_batching=False)

        await pipe.async_batch_apply([1, 2, 3], 0)

        assert sorted(tracker.apply_started) == [1, 2, 3]
        assert tracker.batch_started == []

    @pytest.mark.asyncio
    async def test_batching_enabled_single_timeout_for_all(self):
        """
        With batching enabled: one timeout for the whole batch.
        A batch that takes 100ms with a 200ms timeout succeeds.
        """
        pipe = AsyncTimeoutPipe(
            subpipe=AsyncSlowPipe(delay=0.1),
            timeout=0.3,
            enable_batching=True,
        )
        # The batch processes all items in one 100ms call
        results = await pipe.async_batch_apply([1, 2, 3, 4, 5], 0)
        assert list(results) == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_batching_disabled_per_item_timeout(self):
        """
        With batching disabled: each item gets its own timeout.
        All items running at 100ms with 200ms timeout: all succeed.
        """
        pipe = AsyncTimeoutPipe(
            subpipe=AsyncSlowPipe(delay=0.1),
            timeout=0.3,
            enable_batching=False,
        )
        results = await pipe.async_batch_apply([1, 2, 3, 4, 5], 0)
        assert list(results) == [1, 2, 3, 4, 5]


# ─── Timeout value edge cases ─────────────────────────────────────────────────


class TestAsyncTimeoutPipeTimeoutValues:
    @pytest.mark.asyncio
    async def test_very_short_timeout(self):
        """An extremely short timeout causes even fast-ish operations to fail."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=0.05), timeout=0.001)
        with pytest.raises(TimeoutError):
            await pipe.apply(1, 0)

    @pytest.mark.asyncio
    async def test_very_long_timeout(self):
        """A very long timeout does not affect fast operations."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=3600.0)
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_zero_timeout(self):
        """A zero timeout should immediately timeout any async operation that yields."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=0.01), timeout=0)
        with pytest.raises(TimeoutError):
            await pipe.apply(1, 0)

    @pytest.mark.asyncio
    async def test_timeout_precision(self):
        """The timeout is reasonably precise."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=10.0), timeout=0.1)

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await pipe.apply(1, 0)
        elapsed = time.monotonic() - start

        assert 0.08 <= elapsed <= 0.25


# ─── Multiple sequential calls ────────────────────────────────────────────────


class TestAsyncTimeoutPipeSequentialCalls:
    @pytest.mark.asyncio
    async def test_multiple_successful_calls(self):
        """Multiple sequential calls all succeed."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        for i in range(10):
            result = await pipe.apply(i, 5)
            assert result == i + 5

    @pytest.mark.asyncio
    async def test_timeout_does_not_affect_subsequent_calls(self):
        """After a timeout, subsequent calls still work normally."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=0.5), timeout=0.05)

        with pytest.raises(TimeoutError):
            await pipe.apply(1, 0)

        # Change to a fast pipe scenario: create new pipe with fast subpipe
        fast_pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        result = await fast_pipe.apply(2, 10)
        assert result == 12

    @pytest.mark.asyncio
    async def test_alternating_success_and_timeout(self):
        """
        Alternating between fast and slow operations:
        the timeout state doesn't leak between calls.
        """
        fast = AsyncAddPipe()
        slow = AsyncSlowPipe(delay=1.0)

        fast_pipe = AsyncTimeoutPipe(subpipe=fast, timeout=0.5)
        slow_pipe = AsyncTimeoutPipe(subpipe=slow, timeout=0.05)

        assert await fast_pipe.apply(1, 0) == 1

        with pytest.raises(TimeoutError):
            await slow_pipe.apply(2, 0)

        assert await fast_pipe.apply(3, 0) == 3

        with pytest.raises(TimeoutError):
            await slow_pipe.apply(4, 0)


# ─── Concurrent apply() calls ─────────────────────────────────────────────────


class TestAsyncTimeoutPipeConcurrentApply:
    @pytest.mark.asyncio
    async def test_concurrent_applies_independent_timeouts(self):
        """Each concurrent apply() has its own independent timeout."""
        pipe = AsyncTimeoutPipe(
            subpipe=AsyncVariableDelayPipe(delay_factor=0.1),
            timeout=0.15,
        )

        # Item 1: 0.1s delay (within timeout)
        # Item 3: 0.3s delay (exceeds timeout)
        tasks = [
            asyncio.create_task(pipe.apply(1, 10)),
            asyncio.create_task(pipe.apply(3, 10)),
        ]

        # First task should succeed
        result = await tasks[0]
        assert result == 11

        # Second task should timeout
        with pytest.raises(TimeoutError):
            await tasks[1]

    @pytest.mark.asyncio
    async def test_many_concurrent_calls_within_timeout(self):
        """Many concurrent calls that all finish within timeout succeed."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=0.02), timeout=0.5)

        results = await asyncio.gather(*[pipe.apply(i, 0) for i in range(20)])
        assert list(results) == list(range(20))

    @pytest.mark.asyncio
    async def test_many_concurrent_calls_all_timeout(self):
        """Many concurrent calls that all exceed timeout all fail."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.05)

        with pytest.raises(TimeoutError):
            await asyncio.gather(*[pipe.apply(i, 0) for i in range(10)])


# ─── Graph / visualization tests ──────────────────────────────────────────────


class TestAsyncTimeoutPipeGraph:
    def test_to_graph_returns_valid_graph(self):
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=5.0)
        graph = pipe.to_graph()
        assert graph.is_valid()

    def test_is_wrapper(self):
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=5.0)
        assert pipe.is_wrapper() is True

    def test_get_subgraph_returns_subpipe_graph(self):
        subpipe = AsyncAddPipe()
        pipe = AsyncTimeoutPipe(subpipe=subpipe, timeout=5.0)
        subgraph = pipe.get_subgraph()
        assert subgraph is not None
        assert subgraph.is_valid()

    def test_to_node_includes_timeout_and_batching(self):
        pipe = AsyncTimeoutPipe(
            subpipe=AsyncAddPipe(),
            timeout=3.5,
            enable_batching=True,
            description="My timeout",
        )
        node = pipe.to_node()
        assert "My timeout" in node.title
        assert "timeout=3.5" in node.title
        assert "enable_batching=True" in node.title

    def test_to_node_default_description(self):
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        node = pipe.to_node()
        assert "Cancel after timeout" in node.title

    def test_to_dot_does_not_crash(self):
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        graph = pipe.to_graph()
        dot_str = graph.to_dot()
        assert isinstance(dot_str, str)
        assert len(dot_str) > 0


# ─── Composability tests ──────────────────────────────────────────────────────


class TestAsyncTimeoutPipeComposability:
    @pytest.mark.asyncio
    async def test_chain_with_another_pipe(self):
        """AsyncTimeoutPipe can be chained with other pipes."""

        class AsyncDoublePipe(AsyncPipe[int, int, int]):
            async def apply(self, data: int, metadata: int) -> int:
                return data * 2

        timeout_pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        chain = timeout_pipe | AsyncDoublePipe()

        result = await chain.apply(3, 10)
        # Add: 3+10=13, double: 13*2=26
        assert result == 26

    @pytest.mark.asyncio
    async def test_nested_timeout_pipes(self):
        """A timeout pipe wrapping another timeout pipe uses the shorter timeout."""
        inner = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.5)
        outer = AsyncTimeoutPipe(subpipe=inner, timeout=0.1)

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await outer.apply(1, 0)
        elapsed = time.monotonic() - start

        # Outer timeout (0.1s) fires before inner (0.5s)
        assert elapsed < 0.2

    @pytest.mark.asyncio
    async def test_inner_timeout_fires_first(self):
        """When inner timeout is shorter, it fires first."""
        inner = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.05)
        outer = AsyncTimeoutPipe(subpipe=inner, timeout=1.0)

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await outer.apply(1, 0)
        elapsed = time.monotonic() - start

        # Inner timeout (0.05s) fires before outer (1.0s)
        assert elapsed < 0.2

    @pytest.mark.asyncio
    async def test_used_as_input_to_chain(self):
        """Another pipe can chain into an AsyncTimeoutPipe."""
        first = AsyncAddPipe()
        timeout_pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        chain = first | timeout_pipe

        result = await chain.apply(3, 10)
        # first: 3+10=13, timeout_pipe: 13+10=23
        assert result == 23


# ─── Edge cases ────────────────────────────────────────────────────────────────


class TestAsyncTimeoutPipeEdgeCases:
    @pytest.mark.asyncio
    async def test_frozen_dataclass(self):
        """AsyncTimeoutPipe is immutable."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        with pytest.raises(AttributeError):
            pipe.timeout = 2.0  # type: ignore

    @pytest.mark.asyncio
    async def test_large_batch_within_timeout(self):
        """A large batch that completes quickly succeeds."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0, enable_batching=True)
        data = list(range(1000))
        results = await pipe.async_batch_apply(data, 1)
        assert list(results) == [d + 1 for d in data]

    @pytest.mark.asyncio
    async def test_metadata_passed_correctly(self):
        """Metadata is correctly passed through to the subpipe."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        assert await pipe.apply(5, 100) == 105
        assert await pipe.apply(5, 200) == 205

    @pytest.mark.asyncio
    async def test_string_types(self):
        """AsyncTimeoutPipe works with non-numeric types."""

        class AsyncConcatPipe(AsyncPipe[str, str, str]):
            async def apply(self, data: str, metadata: str) -> str:
                return data + metadata

            async def async_batch_apply(self, data: Sequence[str], metadata: str) -> list[str]:
                return [d + metadata for d in data]

        pipe = AsyncTimeoutPipe(subpipe=AsyncConcatPipe(), timeout=1.0)
        assert await pipe.apply("hello ", "world") == "hello world"

    @pytest.mark.asyncio
    async def test_does_not_leak_between_calls(self):
        """One timed-out call does not affect the state of subsequent calls."""
        subpipe = AsyncSlowPipe(delay=0.5)
        pipe = AsyncTimeoutPipe(subpipe=subpipe, timeout=0.05)

        # This times out
        with pytest.raises(TimeoutError):
            await pipe.apply(1, 0)

        # Create a new pipe with fast subpipe to verify no state leakage
        fast_pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        result = await fast_pipe.apply(42, 0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_default_enable_batching_is_false(self):
        """By default, enable_batching is False."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        assert pipe.enable_batching is False

    @pytest.mark.asyncio
    async def test_batch_apply_base_class_returns_awaitables(self):
        """
        The inherited batch_apply (non-async from BasePipe) returns awaitables
        that can be gathered.
        """
        pipe = AsyncTimeoutPipe(subpipe=AsyncAddPipe(), timeout=1.0)
        awaitables = pipe.batch_apply([1, 2, 3], 10)
        results = await asyncio.gather(*awaitables)
        assert list(results) == [11, 12, 13]


# ─── No deadlock / no hang tests ──────────────────────────────────────────────


class TestAsyncTimeoutPipeNoHang:
    @pytest.mark.asyncio
    async def test_timeout_ensures_bounded_execution(self):
        """The timeout guarantees that apply() always finishes within bounded time."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=100.0), timeout=0.1)

        async with asyncio.timeout(1.0):
            with pytest.raises(TimeoutError):
                await pipe.apply(1, 0)

    @pytest.mark.asyncio
    async def test_batch_timeout_ensures_bounded_execution(self):
        """The timeout guarantees that batch processing finishes within bounded time."""
        pipe = AsyncTimeoutPipe(
            subpipe=AsyncSlowPipe(delay=100.0),
            timeout=0.1,
            enable_batching=True,
        )

        async with asyncio.timeout(1.0):
            with pytest.raises(TimeoutError):
                await pipe.async_batch_apply([1, 2, 3], 0)

    @pytest.mark.asyncio
    async def test_many_timeouts_no_resource_leak(self):
        """Many sequential timeouts don't cause resource issues."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.01)

        async with asyncio.timeout(5.0):
            for _ in range(50):
                with pytest.raises(TimeoutError):
                    await pipe.apply(1, 0)

    @pytest.mark.asyncio
    async def test_concurrent_timeouts_no_hang(self):
        """Many concurrent timeouts all resolve without hanging."""
        pipe = AsyncTimeoutPipe(subpipe=AsyncSlowPipe(delay=1.0), timeout=0.05)

        async with asyncio.timeout(2.0):
            tasks = [pipe.apply(i, 0) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        assert all(isinstance(r, TimeoutError) for r in results)