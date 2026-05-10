import asyncio
from typing import Sequence
import pytest

from kpipeline.async_pipeline import (
    AsyncFallbackPipe,
    AsyncPipe,
)
from kpipeline.pipeline import Pipe, BasePipe
from kpipeline.graph import GraphConnection


# ─── Test fixtures / helper pipes ──────────────────────────────────────────────


class AsyncAddPipe(AsyncPipe[int, int, int]):
    """Async pipe that adds metadata to input."""
    async def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d + metadata for d in data]


class AsyncMultiplyPipe(AsyncPipe[int, int, int]):
    """Async pipe that multiplies input by metadata."""
    async def apply(self, data: int, metadata: int) -> int:
        return data * metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d * metadata for d in data]


class AsyncAlwaysFailPipe(AsyncPipe[int, int, int]):
    """Async pipe that always raises ValueError."""
    async def apply(self, data: int, metadata: int) -> int:
        raise ValueError(f"failed on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise ValueError(f"batch failed on {data}")


class AsyncFailOnNegativePipe(AsyncPipe[int, int, int]):
    """Async pipe that fails on negative inputs."""
    async def apply(self, data: int, metadata: int) -> int:
        if data < 0:
            raise ValueError(f"negative input: {data}")
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        for d in data:
            if d < 0:
                raise ValueError(f"negative input in batch: {d}")
        return [d + metadata for d in data]


class AsyncFailWithTypeErrorPipe(AsyncPipe[int, int, int]):
    """Async pipe that always raises TypeError."""
    async def apply(self, data: int, metadata: int) -> int:
        raise TypeError(f"type error on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise TypeError(f"batch type error on {data}")


class AsyncFailWithKeyErrorPipe(AsyncPipe[int, int, int]):
    """Async pipe that always raises KeyError."""
    async def apply(self, data: int, metadata: int) -> int:
        raise KeyError(f"key error on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise KeyError(f"batch key error on {data}")


class AsyncTrackingPipe(AsyncPipe[int, int, int]):
    """Tracks all calls to apply and async_batch_apply."""
    def __init__(self):
        self.apply_calls: list[tuple[int, int]] = []
        self.batch_apply_calls: list[tuple[list[int], int]] = []

    async def apply(self, data: int, metadata: int) -> int:
        self.apply_calls.append((data, metadata))
        return data

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        self.batch_apply_calls.append((list(data), metadata))
        return list(data)


class AsyncFailingTrackingPipe(AsyncPipe[int, int, int]):
    """Tracks calls then raises."""
    def __init__(self):
        self.apply_calls: list[tuple[int, int]] = []
        self.batch_apply_calls: list[tuple[list[int], int]] = []

    async def apply(self, data: int, metadata: int) -> int:
        self.apply_calls.append((data, metadata))
        raise ValueError(f"tracked failure on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        self.batch_apply_calls.append((list(data), metadata))
        raise ValueError("tracked batch failure")


class AsyncSlowPipe(AsyncPipe[int, int, int]):
    """Async pipe with artificial delay to test concurrency."""
    async def apply(self, data: int, metadata: int) -> int:
        await asyncio.sleep(0.05)
        return data + metadata

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        await asyncio.sleep(0.05)
        return [d + metadata for d in data]


class AsyncSlowFailPipe(AsyncPipe[int, int, int]):
    """Async pipe that takes time then fails."""
    async def apply(self, data: int, metadata: int) -> int:
        await asyncio.sleep(0.05)
        raise ValueError(f"slow fail on {data}")

    async def async_batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        await asyncio.sleep(0.05)
        raise ValueError("slow batch fail")


# Sync pipes for testing mixed sync/async behavior

class SyncAddPipe(Pipe[int, int, int]):
    """Sync pipe that adds metadata to input."""
    def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d + metadata for d in data]


class SyncAlwaysFailPipe(Pipe[int, int, int]):
    """Sync pipe that always raises."""
    def apply(self, data: int, metadata: int) -> int:
        raise ValueError(f"sync failed on {data}")

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise ValueError("sync batch failed")


class SyncMultiplyPipe(Pipe[int, int, int]):
    """Sync pipe that multiplies input by metadata."""
    def apply(self, data: int, metadata: int) -> int:
        return data * metadata

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d * metadata for d in data]


# ─── apply() tests ────────────────────────────────────────────────────────────


class TestAsyncFallbackPipeApply:
    @pytest.mark.asyncio
    async def test_subpipe_succeeds_returns_subpipe_result(self):
        """When the subpipe succeeds, its result is returned directly."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe())
        result = await pipe.apply(3, 10)
        assert result == 13  # 3 + 10

    @pytest.mark.asyncio
    async def test_subpipe_fails_returns_fallback_result(self):
        """When the subpipe raises a matching exception, the fallback is used."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe())
        result = await pipe.apply(3, 10)
        assert result == 13  # fallback: 3 + 10

    @pytest.mark.asyncio
    async def test_fallback_receives_original_input(self):
        """The fallback pipe receives the same input and metadata as the subpipe."""
        subpipe = AsyncFailingTrackingPipe()
        fallback = AsyncTrackingPipe()
        pipe = AsyncFallbackPipe(subpipe=subpipe, fallback=fallback)

        await pipe.apply(42, 7)

        assert subpipe.apply_calls == [(42, 7)]
        assert fallback.apply_calls == [(42, 7)]

    @pytest.mark.asyncio
    async def test_subpipe_succeeds_fallback_not_called(self):
        """When the subpipe succeeds, the fallback is never invoked."""
        fallback = AsyncTrackingPipe()
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=fallback)

        await pipe.apply(1, 2)

        assert fallback.apply_calls == []
        assert fallback.batch_apply_calls == []

    @pytest.mark.asyncio
    async def test_conditional_failure_only_uses_fallback_on_failure(self):
        """Fallback is only used for items that actually fail."""
        pipe = AsyncFallbackPipe(subpipe=AsyncFailOnNegativePipe(), fallback=AsyncMultiplyPipe())

        assert await pipe.apply(5, 10) == 15   # subpipe: 5 + 10
        assert await pipe.apply(-3, 10) == -30  # fallback: -3 * 10

    @pytest.mark.asyncio
    async def test_apply_awaits_async_subpipe(self):
        """apply() correctly awaits async subpipe results."""
        pipe = AsyncFallbackPipe(subpipe=AsyncSlowPipe(), fallback=AsyncAddPipe())
        result = await pipe.apply(5, 10)
        assert result == 15

    @pytest.mark.asyncio
    async def test_apply_awaits_async_fallback(self):
        """apply() correctly awaits async fallback results."""
        pipe = AsyncFallbackPipe(subpipe=AsyncSlowFailPipe(), fallback=AsyncSlowPipe())
        result = await pipe.apply(5, 10)
        assert result == 15


# ─── Exception filtering tests ────────────────────────────────────────────────


class TestAsyncFallbackPipeExceptionFiltering:
    @pytest.mark.asyncio
    async def test_matching_single_exception_type_caught(self):
        """A single exception type in `exceptions` is caught correctly."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=AsyncAddPipe(),
            exceptions=ValueError,
        )
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_non_matching_exception_type_propagates(self):
        """An exception not in `exceptions` is re-raised."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailWithTypeErrorPipe(),
            fallback=AsyncAddPipe(),
            exceptions=ValueError,
        )
        with pytest.raises(TypeError, match="type error on 5"):
            await pipe.apply(5, 10)

    @pytest.mark.asyncio
    async def test_matching_tuple_of_exception_types(self):
        """A tuple of exception types catches any matching exception."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailWithTypeErrorPipe(),
            fallback=AsyncAddPipe(),
            exceptions=(ValueError, TypeError),
        )
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_non_matching_with_tuple_of_exception_types(self):
        """An exception not in the tuple is re-raised."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailWithKeyErrorPipe(),
            fallback=AsyncAddPipe(),
            exceptions=(ValueError, TypeError),
        )
        with pytest.raises(KeyError):
            await pipe.apply(3, 10)

    @pytest.mark.asyncio
    async def test_default_exceptions_catches_all(self):
        """The default `exceptions=Exception` catches any standard exception."""
        for subpipe in [AsyncAlwaysFailPipe(), AsyncFailWithTypeErrorPipe(), AsyncFailWithKeyErrorPipe()]:
            pipe = AsyncFallbackPipe(subpipe=subpipe, fallback=AsyncAddPipe())
            result = await pipe.apply(1, 2)
            assert result == 3

    @pytest.mark.asyncio
    async def test_exception_subclass_is_caught(self):
        """A subclass of the specified exception type is caught."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=AsyncAddPipe(),
            exceptions=Exception,
        )
        result = await pipe.apply(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_fallback_itself_raises(self):
        """If the fallback also fails, the fallback's exception propagates."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=AsyncFailWithTypeErrorPipe(),
        )
        with pytest.raises(TypeError, match="type error on"):
            await pipe.apply(5, 10)


# ─── async_batch_apply with enable_batching=True ───────────────────────────────


class TestAsyncFallbackPipeBatchingEnabled:
    @pytest.mark.asyncio
    async def test_subpipe_batch_succeeds(self):
        """When subpipe batch succeeds, results are returned directly."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), enable_batching=True)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_subpipe_batch_fails_uses_fallback_batch(self):
        """When subpipe batch fails, entire batch goes to fallback."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe(), enable_batching=True)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_single_bad_item_fails_entire_batch(self):
        """With batching enabled, one bad item causes the whole batch to go to fallback."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailOnNegativePipe(),
            fallback=AsyncMultiplyPipe(),
            enable_batching=True,
        )
        results = await pipe.async_batch_apply([1, -2, 3], 10)
        assert list(results) == [10, -20, 30]  # all multiplied

    @pytest.mark.asyncio
    async def test_batch_uses_async_batch_apply_not_individual_apply(self):
        """With batching enabled, subpipe's batch method is called."""
        subpipe = AsyncFailingTrackingPipe()
        fallback = AsyncTrackingPipe()
        pipe = AsyncFallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=True)

        await pipe.async_batch_apply([1, 2], 0)

        assert subpipe.batch_apply_calls == [([1, 2], 0)]
        assert subpipe.apply_calls == []
        assert fallback.batch_apply_calls == [([1, 2], 0)]
        assert fallback.apply_calls == []

    @pytest.mark.asyncio
    async def test_batch_exception_filtering(self):
        """With batching, exception filtering applies to batch_apply too."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailWithTypeErrorPipe(),
            fallback=AsyncAddPipe(),
            exceptions=ValueError,
            enable_batching=True,
        )
        with pytest.raises(TypeError):
            await pipe.async_batch_apply([1, 2], 10)

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """An empty batch should work with batching enabled."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), enable_batching=True)
        results = await pipe.async_batch_apply([], 10)
        assert list(results) == []


# ─── async_batch_apply with enable_batching=False ──────────────────────────────


class TestAsyncFallbackPipeBatchingDisabled:
    @pytest.mark.asyncio
    async def test_subpipe_all_succeed(self):
        """When all items succeed, results come from the subpipe."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), enable_batching=False)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_single_bad_item_only_that_item_uses_fallback(self):
        """With batching disabled, only failing items use the fallback."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailOnNegativePipe(),
            fallback=AsyncMultiplyPipe(),
            enable_batching=False,
        )
        results = await pipe.async_batch_apply([1, -2, 3], 10)
        # 1 succeeds (1+10=11), -2 fails → fallback (-2*10=-20), 3 succeeds (3+10=13)
        assert list(results) == [11, -20, 13]

    @pytest.mark.asyncio
    async def test_all_items_fail(self):
        """When all items fail, all go to fallback."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe(), enable_batching=False)
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_items_processed_concurrently(self):
        """With batching disabled, items run concurrently via asyncio.gather."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncSlowPipe(),  # 50ms per item
            fallback=AsyncAddPipe(),
            enable_batching=False,
        )

        import time
        start = time.monotonic()
        results = await pipe.async_batch_apply([1, 2, 3, 4, 5], 10)
        elapsed = time.monotonic() - start

        assert list(results) == [11, 12, 13, 14, 15]
        # If sequential, would take 5 * 50ms = 250ms
        # If concurrent, should take ~50ms
        assert elapsed < 0.2, f"Expected concurrent execution, took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_concurrent_fallback_execution(self):
        """When items fail, fallbacks also run concurrently."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncSlowFailPipe(),  # 50ms then fail
            fallback=AsyncSlowPipe(),     # 50ms then succeed
            enable_batching=False,
        )

        import time
        start = time.monotonic()
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        elapsed = time.monotonic() - start

        assert list(results) == [11, 12, 13]
        # Each item: 50ms fail + 50ms fallback = 100ms sequential per item
        # But items are concurrent, so total should be ~100ms, not 300ms
        assert elapsed < 0.25, f"Expected concurrent execution, took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_exception_filtering_per_item(self):
        """With batching disabled, exception filtering applies per item."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailWithTypeErrorPipe(),
            fallback=AsyncAddPipe(),
            exceptions=ValueError,
            enable_batching=False,
        )
        with pytest.raises(TypeError):
            await pipe.async_batch_apply([1], 10)

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """An empty batch should work with batching disabled."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), enable_batching=False)
        results = await pipe.async_batch_apply([], 10)
        assert list(results) == []

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure_preserves_order(self):
        """Results maintain input order regardless of which items fail."""

        class AsyncFailOnEvenPipe(AsyncPipe[int, int, int]):
            async def apply(self, data: int, metadata: int) -> int:
                if data % 2 == 0:
                    raise ValueError(f"even: {data}")
                return data * 100  # distinct from fallback

        pipe = AsyncFallbackPipe(
            subpipe=AsyncFailOnEvenPipe(),
            fallback=AsyncAddPipe(),  # returns data + metadata
            enable_batching=False,
        )
        results = await pipe.async_batch_apply([1, 2, 3, 4, 5], 1)
        # odd: 1*100=100, even: 2+1=3, odd: 3*100=300, even: 4+1=5, odd: 5*100=500
        assert list(results) == [100, 3, 300, 5, 500]


# ─── Comparison: batching enabled vs disabled ──────────────────────────────────


class TestAsyncFallbackPipeBatchingComparison:
    @pytest.mark.asyncio
    async def test_partial_failure_different_behavior(self):
        """
        Key behavioral difference: with batching enabled, one failure contaminates
        the whole batch. With batching disabled, only the failing item uses fallback.
        """
        data = [5, -1, 3]
        meta = 2

        batched = AsyncFallbackPipe(
            subpipe=AsyncFailOnNegativePipe(),
            fallback=AsyncMultiplyPipe(),
            enable_batching=True,
        )
        unbatched = AsyncFallbackPipe(
            subpipe=AsyncFailOnNegativePipe(),
            fallback=AsyncMultiplyPipe(),
            enable_batching=False,
        )

        batched_results = list(await batched.async_batch_apply(data, meta))
        unbatched_results = list(await unbatched.async_batch_apply(data, meta))

        # Batched: entire batch fails → all go to fallback (multiply)
        assert batched_results == [10, -2, 6]
        # Unbatched: only -1 fails → mixed results
        assert unbatched_results == [7, -2, 5]
        # They differ!
        assert batched_results != unbatched_results

    @pytest.mark.asyncio
    async def test_no_failure_same_behavior(self):
        """When nothing fails, both modes produce the same results."""
        data = [1, 2, 3]
        meta = 10

        batched = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), enable_batching=True)
        unbatched = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), enable_batching=False)

        assert list(await batched.async_batch_apply(data, meta)) == list(await unbatched.async_batch_apply(data, meta))


# ─── Mixed sync/async subpipes ────────────────────────────────────────────────


class TestAsyncFallbackPipeMixedSyncAsync:
    @pytest.mark.asyncio
    async def test_sync_subpipe_async_fallback(self):
        """AsyncFallbackPipe works with a sync subpipe and async fallback."""
        pipe = AsyncFallbackPipe(
            subpipe=SyncAlwaysFailPipe(),
            fallback=AsyncAddPipe(),
        )
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_async_subpipe_sync_fallback(self):
        """AsyncFallbackPipe works with an async subpipe and sync fallback."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=SyncAddPipe(),
        )
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_sync_subpipe_sync_fallback(self):
        """AsyncFallbackPipe works with both sync subpipe and sync fallback."""
        pipe = AsyncFallbackPipe(
            subpipe=SyncAlwaysFailPipe(),
            fallback=SyncAddPipe(),
        )
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_sync_subpipe_succeeds(self):
        """AsyncFallbackPipe correctly handles sync subpipe success."""
        pipe = AsyncFallbackPipe(
            subpipe=SyncAddPipe(),
            fallback=AsyncMultiplyPipe(),
        )
        result = await pipe.apply(3, 10)
        assert result == 13

    @pytest.mark.asyncio
    async def test_batch_with_sync_subpipe_and_async_fallback(self):
        """Batching works with mixed sync/async pipes."""
        pipe = AsyncFallbackPipe(
            subpipe=SyncAlwaysFailPipe(),
            fallback=AsyncAddPipe(),
            enable_batching=True,
        )
        results = await pipe.async_batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_batch_disabled_with_sync_subpipe(self):
        """Non-batched mode works with sync subpipes."""

        class SyncFailOnNegativePipe(Pipe[int, int, int]):
            def apply(self, data: int, metadata: int) -> int:
                if data < 0:
                    raise ValueError("negative")
                return data + metadata

        pipe = AsyncFallbackPipe(
            subpipe=SyncFailOnNegativePipe(),
            fallback=AsyncMultiplyPipe(),
            enable_batching=False,
        )
        results = await pipe.async_batch_apply([1, -2, 3], 10)
        assert list(results) == [11, -20, 13]


# ─── Error handling edge cases ─────────────────────────────────────────────────


class TestAsyncFallbackPipeErrorEdgeCases:
    @pytest.mark.asyncio
    async def test_fallback_also_raises_matching_exception(self):
        """If fallback raises, its exception propagates (not caught again)."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=AsyncAlwaysFailPipe(),
            exceptions=ValueError,
        )
        with pytest.raises(ValueError, match="failed on"):
            await pipe.apply(5, 10)

    @pytest.mark.asyncio
    async def test_fallback_raises_different_exception(self):
        """Fallback's exception propagates regardless of the exceptions filter."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=AsyncFailWithTypeErrorPipe(),
            exceptions=ValueError,
        )
        with pytest.raises(TypeError, match="type error on"):
            await pipe.apply(5, 10)

    @pytest.mark.asyncio
    async def test_batch_fallback_raises(self):
        """If fallback's batch_apply also raises, the exception propagates."""
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=AsyncFailWithTypeErrorPipe(),
            enable_batching=True,
        )
        with pytest.raises(TypeError):
            await pipe.async_batch_apply([1, 2], 10)

    @pytest.mark.asyncio
    async def test_non_batched_partial_fallback_failure(self):
        """
        With batching disabled, if some items' fallbacks fail,
        the entire gather raises (first exception).
        """

        class AsyncFailOnSpecificPipe(AsyncPipe[int, int, int]):
            async def apply(self, data: int, metadata: int) -> int:
                if data == 99:
                    raise TypeError("fallback failed on 99")
                return data + metadata

        pipe = AsyncFallbackPipe(
            subpipe=AsyncAlwaysFailPipe(),
            fallback=AsyncFailOnSpecificPipe(),
            enable_batching=False,
        )

        # Item 99 will fail in both subpipe and fallback
        with pytest.raises(TypeError, match="fallback failed on 99"):
            await pipe.async_batch_apply([1, 99, 3], 10)


# ─── Graph / visualization tests ──────────────────────────────────────────────


class TestAsyncFallbackPipeGraph:
    def test_to_graph_returns_valid_graph(self):
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), description="Test fallback")
        graph = pipe.to_graph()
        assert graph.is_valid()

    def test_to_graph_has_correct_inputs(self):
        """Graph inputs come from the subpipe path, not the fallback."""
        subpipe = AsyncAddPipe()
        fallback = AsyncMultiplyPipe()
        pipe = AsyncFallbackPipe(subpipe=subpipe, fallback=fallback)
        graph = pipe.to_graph()

        fallback_graph = fallback.to_graph()
        # Fallback inputs should NOT be top-level graph inputs
        for fallback_input in fallback_graph.inputs:
            if fallback_input in graph.inputs:
                # It's acceptable only if it's also a subpipe input
                subpipe_graph = subpipe.to_graph()
                assert fallback_input in subpipe_graph.inputs

    def test_to_graph_contains_on_failure_connection(self):
        """The graph should have a connection labeled 'On failure' to the fallback."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe())
        graph = pipe.to_graph()

        on_failure_connections = [c for c in graph.connections if c.label == "On failure"]
        assert len(on_failure_connections) > 0

    def test_to_node_includes_description_and_batching_flag(self):
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAddPipe(),
            fallback=AsyncMultiplyPipe(),
            description="My async fallback",
            enable_batching=False,
        )
        node = pipe.to_node()
        assert "My async fallback" in node.title
        assert "enable_batching=False" in node.title

    def test_to_node_batching_enabled(self):
        pipe = AsyncFallbackPipe(
            subpipe=AsyncAddPipe(),
            fallback=AsyncMultiplyPipe(),
            description="Batched",
            enable_batching=True,
        )
        node = pipe.to_node()
        assert "enable_batching=True" in node.title

    def test_is_wrapper(self):
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe())
        assert pipe.is_wrapper() is True

    def test_get_subgraph_returns_subpipe_graph(self):
        subpipe = AsyncAddPipe()
        pipe = AsyncFallbackPipe(subpipe=subpipe, fallback=AsyncMultiplyPipe())
        subgraph = pipe.get_subgraph()
        assert subgraph is not None
        assert subgraph.is_valid()

    def test_to_dot_does_not_crash(self):
        """Smoke test: DOT export should not raise."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe(), description="DOT test")
        graph = pipe.to_graph()
        dot_str = graph.to_dot()
        assert isinstance(dot_str, str)
        assert len(dot_str) > 0


# ─── Composability tests ──────────────────────────────────────────────────────


class TestAsyncFallbackPipeComposability:
    @pytest.mark.asyncio
    async def test_chain_with_async_pipe(self):
        """AsyncFallbackPipe can be chained with another async pipe using |."""
        fallback_pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe())
        chain = fallback_pipe | AsyncMultiplyPipe()

        result = await chain.apply(3, 10)
        # fallback: 3+10=13, then multiply: 13*10=130
        assert result == 130

    @pytest.mark.asyncio
    async def test_chain_with_sync_pipe(self):
        """AsyncFallbackPipe can be chained with a sync pipe."""
        fallback_pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe())
        chain = fallback_pipe | SyncMultiplyPipe()

        result = await chain.apply(3, 10)
        assert result == 130

    @pytest.mark.asyncio
    async def test_nested_async_fallback(self):
        """An AsyncFallbackPipe can be nested inside another."""
        inner = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe())
        outer = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=inner)

        result = await outer.apply(3, 10)
        # outer subpipe fails → outer fallback (inner) is used
        # inner subpipe fails → inner fallback (AsyncAddPipe) is used
        assert result == 13

    @pytest.mark.asyncio
    async def test_as_subpipe_of_another_fallback(self):
        """An AsyncFallbackPipe can be the subpipe of another fallback."""
        inner = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAlwaysFailPipe())
        outer = AsyncFallbackPipe(subpipe=inner, fallback=AsyncAddPipe())

        result = await outer.apply(3, 10)
        # inner: both fail → raises
        # outer catches → uses fallback
        assert result == 13

    @pytest.mark.asyncio
    async def test_used_as_input_to_chain(self):
        """Another pipe can chain into an AsyncFallbackPipe."""
        first = AsyncAddPipe()
        fallback_pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncMultiplyPipe())
        chain = first | fallback_pipe

        result = await chain.apply(3, 10)
        # first: 3+10=13, then fallback subpipe fails → fallback: 13*10=130
        assert result == 130


# ─── Edge cases ────────────────────────────────────────────────────────────────


class TestAsyncFallbackPipeEdgeCases:
    @pytest.mark.asyncio
    async def test_frozen_dataclass(self):
        """AsyncFallbackPipe is immutable."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe())
        with pytest.raises(AttributeError):
            pipe.subpipe = AsyncMultiplyPipe()  # type: ignore

    @pytest.mark.asyncio
    async def test_single_item_batch(self):
        """Batch of one item works correctly."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe())
        results = await pipe.async_batch_apply([5], 10)
        assert list(results) == [15]

    @pytest.mark.asyncio
    async def test_large_batch(self):
        """Large batches work correctly."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAddPipe(), fallback=AsyncMultiplyPipe())
        data = list(range(1000))
        results = await pipe.async_batch_apply(data, 1)
        assert list(results) == [d + 1 for d in data]

    @pytest.mark.asyncio
    async def test_large_batch_all_failing(self):
        """Large batches with all items failing works correctly."""
        pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe(), enable_batching=False)
        data = list(range(100))
        results = await pipe.async_batch_apply(data, 1)
        assert list(results) == [d + 1 for d in data]

    @pytest.mark.asyncio
    async def test_string_types(self):
        """AsyncFallbackPipe works with non-numeric types."""

        class AsyncConcatPipe(AsyncPipe[str, str, str]):
            async def apply(self, data: str, metadata: str) -> str:
                return data + metadata

        class AsyncFailStringPipe(AsyncPipe[str, str, str]):
            async def apply(self, data: str, metadata: str) -> str:
                raise ValueError("nope")

        pipe = AsyncFallbackPipe(subpipe=AsyncFailStringPipe(), fallback=AsyncConcatPipe())
        assert await pipe.apply("hello ", "world") == "hello world"

    @pytest.mark.asyncio
    async def test_does_not_deadlock(self):
        """Ensure no deadlock under normal usage with a timeout."""
        pipe = AsyncFallbackPipe(subpipe=AsyncSlowFailPipe(), fallback=AsyncSlowPipe())

        async with asyncio.timeout(2.0):
            results = await pipe.async_batch_apply([1, 2, 3], 10)

        assert list(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_default_batch_apply_via_base_class(self):
        """
        The inherited batch_apply from BasePipe (non-async) calls apply() in sequence.
        This tests that the base class method still works.
        """
        pipe = AsyncFallbackPipe(subpipe=AsyncAlwaysFailPipe(), fallback=AsyncAddPipe())
        # batch_apply (the sync wrapper from BasePipe) returns Sequence[Awaitable[Output]]
        awaitables = pipe.batch_apply([1, 2, 3], 10)
        results = await asyncio.gather(*awaitables)
        assert list(results) == [11, 12, 13]
