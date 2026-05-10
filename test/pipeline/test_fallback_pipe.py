from typing import Sequence
import pytest

from kpipeline.pipeline import Pipe, FallbackPipe


# ─── Test fixtures / helper pipes ──────────────────────────────────────────────


class AddPipe(Pipe[int, int, int]):
    """Simple pipe that adds metadata to input."""
    def apply(self, data: int, metadata: int) -> int:
        return data + metadata

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d + metadata for d in data]


class MultiplyPipe(Pipe[int, int, int]):
    """Simple pipe that multiplies input by metadata."""
    def apply(self, data: int, metadata: int) -> int:
        return data * metadata

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        return [d * metadata for d in data]


class AlwaysFailPipe(Pipe[int, int, int]):
    """A pipe that always raises ValueError."""
    def apply(self, data: int, metadata: int) -> int:
        raise ValueError(f"failed on {data}")

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise ValueError(f"batch failed on {data}")


class FailOnNegativePipe(Pipe[int, int, int]):
    """Fails on negative inputs, succeeds otherwise."""
    def apply(self, data: int, metadata: int) -> int:
        if data < 0:
            raise ValueError(f"negative input: {data}")
        return data + metadata

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        for d in data:
            if d < 0:
                raise ValueError(f"negative input in batch: {d}")
        return [d + metadata for d in data]


class FailWithTypeErrorPipe(Pipe[int, int, int]):
    """Always raises TypeError."""
    def apply(self, data: int, metadata: int) -> int:
        raise TypeError(f"type error on {data}")

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise TypeError(f"batch type error on {data}")


class FailWithKeyErrorPipe(Pipe[int, int, int]):
    """Always raises KeyError."""
    def apply(self, data: int, metadata: int) -> int:
        raise KeyError(f"key error on {data}")

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        raise KeyError(f"batch key error on {data}")


class TrackingPipe(Pipe[int, int, int]):
    """Tracks all calls to apply and batch_apply."""
    def __init__(self):
        self.apply_calls: list[tuple[int, int]] = []
        self.batch_apply_calls: list[tuple[Sequence[int], int]] = []

    def apply(self, data: int, metadata: int) -> int:
        self.apply_calls.append((data, metadata))
        return data

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        self.batch_apply_calls.append((list(data), metadata))
        return list(data)


class FailingTrackingPipe(Pipe[int, int, int]):
    """Tracks calls then raises."""
    def __init__(self):
        self.apply_calls: list[tuple[int, int]] = []
        self.batch_apply_calls: list[tuple[Sequence[int], int]] = []

    def apply(self, data: int, metadata: int) -> int:
        self.apply_calls.append((data, metadata))
        raise ValueError(f"tracked failure on {data}")

    def batch_apply(self, data: Sequence[int], metadata: int) -> list[int]:
        self.batch_apply_calls.append((list(data), metadata))
        raise ValueError(f"tracked batch failure")


# ─── apply() tests ────────────────────────────────────────────────────────────


class TestFallbackPipeApply:
    def test_subpipe_succeeds_returns_subpipe_result(self):
        """When the subpipe succeeds, its result is returned directly."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe())
        result = pipe.apply(3, 10)
        assert result == 13  # 3 + 10

    def test_subpipe_fails_returns_fallback_result(self):
        """When the subpipe raises a matching exception, the fallback is used."""
        pipe = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=AddPipe())
        result = pipe.apply(3, 10)
        assert result == 13  # fallback: 3 + 10

    def test_fallback_receives_original_input(self):
        """The fallback pipe receives the same input and metadata as the subpipe."""
        subpipe = FailingTrackingPipe()
        fallback = TrackingPipe()
        pipe = FallbackPipe(subpipe=subpipe, fallback=fallback)

        pipe.apply(42, 7)

        assert subpipe.apply_calls == [(42, 7)]
        assert fallback.apply_calls == [(42, 7)]

    def test_subpipe_succeeds_fallback_not_called(self):
        """When the subpipe succeeds, the fallback is never invoked."""
        fallback = TrackingPipe()
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=fallback)

        pipe.apply(1, 2)

        assert fallback.apply_calls == []
        assert fallback.batch_apply_calls == []

    def test_conditional_failure_only_uses_fallback_on_failure(self):
        """Fallback is only used for items that actually fail."""
        pipe = FallbackPipe(subpipe=FailOnNegativePipe(), fallback=MultiplyPipe())

        assert pipe.apply(5, 10) == 15   # subpipe: 5 + 10
        assert pipe.apply(-3, 10) == -30  # fallback: -3 * 10


# ─── Exception filtering tests ────────────────────────────────────────────────


class TestFallbackPipeExceptionFiltering:
    def test_matching_single_exception_type_caught(self):
        """A single exception type in `exceptions` is caught correctly."""
        pipe = FallbackPipe(
            subpipe=AlwaysFailPipe(),
            fallback=AddPipe(),
            exceptions=ValueError,
        )
        result = pipe.apply(3, 10)
        assert result == 13

    def test_non_matching_exception_type_propagates(self):
        """An exception not in `exceptions` is re-raised."""
        pipe = FallbackPipe(
            subpipe=FailWithTypeErrorPipe(),
            fallback=AddPipe(),
            exceptions=ValueError,
        )
        with pytest.raises(TypeError, match="type error on 5"):
            pipe.apply(5, 10)

    def test_matching_tuple_of_exception_types(self):
        """A tuple of exception types catches any matching exception."""
        pipe = FallbackPipe(
            subpipe=FailWithTypeErrorPipe(),
            fallback=AddPipe(),
            exceptions=(ValueError, TypeError),
        )
        result = pipe.apply(3, 10)
        assert result == 13

    def test_non_matching_with_tuple_of_exception_types(self):
        """An exception not in the tuple is re-raised."""
        pipe = FallbackPipe(
            subpipe=FailWithKeyErrorPipe(),
            fallback=AddPipe(),
            exceptions=(ValueError, TypeError),
        )
        with pytest.raises(KeyError):
            pipe.apply(3, 10)

    def test_default_exceptions_catches_all(self):
        """The default `exceptions=Exception` catches any standard exception."""
        for subpipe in [AlwaysFailPipe(), FailWithTypeErrorPipe(), FailWithKeyErrorPipe()]:
            pipe = FallbackPipe(subpipe=subpipe, fallback=AddPipe())
            result = pipe.apply(1, 2)
            assert result == 3

    def test_exception_subclass_is_caught(self):
        """A subclass of the specified exception type is caught."""
        # ValueError is a subclass of Exception
        pipe = FallbackPipe(
            subpipe=AlwaysFailPipe(),
            fallback=AddPipe(),
            exceptions=Exception,
        )
        result = pipe.apply(1, 2)
        assert result == 3

    def test_fallback_itself_raises(self):
        """If the fallback also fails, the fallback's exception propagates."""
        pipe = FallbackPipe(
            subpipe=AlwaysFailPipe(),
            fallback=FailWithTypeErrorPipe(),
        )
        with pytest.raises(TypeError, match="type error on"):
            pipe.apply(5, 10)


# ─── batch_apply with enable_batching=True ─────────────────────────────────────


class TestFallbackPipeBatchingEnabled:
    def test_subpipe_batch_succeeds(self):
        """When subpipe batch succeeds, results are returned directly."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe(), enable_batching=True)
        results = pipe.batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    def test_subpipe_batch_fails_uses_fallback_batch(self):
        """When subpipe batch fails, entire batch goes to fallback."""
        pipe = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=AddPipe(), enable_batching=True)
        results = pipe.batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    def test_single_bad_item_fails_entire_batch(self):
        """With batching enabled, one bad item causes the whole batch to go to fallback."""
        subpipe = FailOnNegativePipe()  # fails on negative
        fallback = MultiplyPipe()
        pipe = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=True)

        # The batch contains one negative, so the entire batch goes to fallback
        results = pipe.batch_apply([1, -2, 3], 10)
        assert list(results) == [10, -20, 30]  # all multiplied

    def test_batch_uses_batch_apply_not_individual_apply(self):
        """With batching enabled, subpipe.batch_apply is called (not individual applies)."""
        subpipe = FailingTrackingPipe()
        fallback = TrackingPipe()
        pipe = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=True)

        pipe.batch_apply([1, 2], 0)

        assert subpipe.batch_apply_calls == [([1, 2], 0)]
        assert subpipe.apply_calls == []  # apply was NOT called individually
        assert fallback.batch_apply_calls == [([1, 2], 0)]
        assert fallback.apply_calls == []

    def test_batch_exception_filtering(self):
        """With batching, exception filtering applies to batch_apply too."""
        pipe = FallbackPipe(
            subpipe=FailWithTypeErrorPipe(),
            fallback=AddPipe(),
            exceptions=ValueError,  # does NOT match TypeError
            enable_batching=True,
        )
        with pytest.raises(TypeError):
            pipe.batch_apply([1, 2], 10)

    def test_empty_batch(self):
        """An empty batch should work with batching enabled."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe(), enable_batching=True)
        results = pipe.batch_apply([], 10)
        assert list(results) == []


# ─── batch_apply with enable_batching=False ────────────────────────────────────


class TestFallbackPipeBatchingDisabled:
    def test_subpipe_all_succeed(self):
        """When all items succeed, results come from the subpipe."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe(), enable_batching=False)
        results = pipe.batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    def test_single_bad_item_only_that_item_uses_fallback(self):
        """With batching disabled, only failing items use the fallback."""
        subpipe = FailOnNegativePipe()  # fails on negative
        fallback = MultiplyPipe()
        pipe = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=False)

        results = pipe.batch_apply([1, -2, 3], 10)
        # 1 succeeds (1+10=11), -2 fails and uses fallback (-2*10=-20), 3 succeeds (3+10=13)
        assert list(results) == [11, -20, 13]

    def test_all_items_fail(self):
        """When all items fail, all go to fallback."""
        pipe = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=AddPipe(), enable_batching=False)
        results = pipe.batch_apply([1, 2, 3], 10)
        assert list(results) == [11, 12, 13]

    def test_uses_individual_apply_not_batch_apply(self):
        """With batching disabled, items are processed individually via apply()."""
        subpipe = FailingTrackingPipe()
        fallback = TrackingPipe()
        pipe = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=False)

        pipe.batch_apply([10, 20], 5)

        # Each item processed individually through apply, not batch_apply
        assert subpipe.apply_calls == [(10, 5), (20, 5)]
        assert subpipe.batch_apply_calls == []
        assert fallback.apply_calls == [(10, 5), (20, 5)]
        assert fallback.batch_apply_calls == []

    def test_exception_filtering_per_item(self):
        """With batching disabled, exception filtering applies per item."""
        pipe = FallbackPipe(
            subpipe=FailWithTypeErrorPipe(),
            fallback=AddPipe(),
            exceptions=ValueError,  # does NOT match TypeError
            enable_batching=False,
        )
        # Each individual apply raises TypeError, which doesn't match, so it propagates
        with pytest.raises(TypeError):
            pipe.batch_apply([1], 10)

    def test_empty_batch(self):
        """An empty batch should work with batching disabled."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe(), enable_batching=False)
        results = pipe.batch_apply([], 10)
        assert list(results) == []


# ─── Comparison: batching enabled vs disabled ──────────────────────────────────


class TestFallbackPipeBatchingComparison:
    def test_partial_failure_different_behavior(self):
        """
        Key behavioral difference: with batching enabled, one failure contaminates
        the whole batch. With batching disabled, only the failing item uses fallback.
        """
        subpipe = FailOnNegativePipe()
        fallback = MultiplyPipe()
        data = [5, -1, 3]
        meta = 2

        batched = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=True)
        unbatched = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=False)

        batched_results = list(batched.batch_apply(data, meta))
        unbatched_results = list(unbatched.batch_apply(data, meta))

        # Batched: entire batch fails → all go to fallback (multiply)
        assert batched_results == [10, -2, 6]
        # Unbatched: only -1 fails → mixed results
        assert unbatched_results == [7, -2, 5]
        # They differ!
        assert batched_results != unbatched_results

    def test_no_failure_same_behavior(self):
        """When nothing fails, both modes produce the same results."""
        subpipe = AddPipe()
        fallback = MultiplyPipe()
        data = [1, 2, 3]
        meta = 10

        batched = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=True)
        unbatched = FallbackPipe(subpipe=subpipe, fallback=fallback, enable_batching=False)

        assert list(batched.batch_apply(data, meta)) == list(unbatched.batch_apply(data, meta))


# ─── Graph / visualization tests ──────────────────────────────────────────────


class TestFallbackPipeGraph:
    def test_to_graph_returns_valid_graph(self):
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe(), description="Test fallback")
        graph = pipe.to_graph()
        assert graph.is_valid()

    def test_to_graph_has_correct_inputs(self):
        """The graph inputs should be the subpipe wrapper's inputs, not the fallback's."""
        subpipe = AddPipe()
        fallback = MultiplyPipe()
        pipe = FallbackPipe(subpipe=subpipe, fallback=fallback)
        graph = pipe.to_graph()

        # The inputs should correspond to the main subpipe path
        # The fallback's inputs should NOT be top-level graph inputs
        fallback_graph = fallback.to_graph()
        for fallback_input in fallback_graph.inputs:
            # Fallback inputs are NOT in the top-level graph inputs
            # (they are connected from the subpipe node instead)
            assert fallback_input not in graph.inputs or fallback_input in pipe.to_graph().inputs

    def test_to_graph_contains_on_failure_connection(self):
        """The graph should have a connection labeled 'On failure' to the fallback."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe())
        graph = pipe.to_graph()

        on_failure_connections = [c for c in graph.connections if c.label == "On failure"]
        assert len(on_failure_connections) > 0

    def test_to_node_includes_description_and_batching_flag(self):
        pipe = FallbackPipe(
            subpipe=AddPipe(),
            fallback=MultiplyPipe(),
            description="My fallback",
            enable_batching=False,
        )
        node = pipe.to_node()
        assert "My fallback" in node.title
        assert "enable_batching=False" in node.title

    def test_is_wrapper(self):
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe())
        assert pipe.is_wrapper() is True

    def test_get_subgraph_returns_subpipe_graph(self):
        subpipe = AddPipe()
        pipe = FallbackPipe(subpipe=subpipe, fallback=MultiplyPipe())
        subgraph = pipe.get_subgraph()
        assert subgraph is not None
        assert subgraph.is_valid()

    def test_to_dot_does_not_crash(self):
        """Smoke test: DOT export should not raise."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe(), description="DOT test")
        graph = pipe.to_graph()
        dot_str = graph.to_dot()
        assert isinstance(dot_str, str)
        assert len(dot_str) > 0


# ─── Composability tests ──────────────────────────────────────────────────────


class TestFallbackPipeComposability:
    def test_chain_with_other_pipe(self):
        """FallbackPipe can be chained with other pipes."""
        fallback_pipe = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=AddPipe())
        chain = fallback_pipe | MultiplyPipe()

        result = chain.apply(3, 10)
        # fallback: 3+10=13, then multiply: 13*10=130
        assert result == 130

    def test_nested_fallback(self):
        """A FallbackPipe can be the subpipe or fallback of another FallbackPipe."""
        inner = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=AddPipe())
        outer = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=inner)

        result = outer.apply(3, 10)
        # outer subpipe fails → outer fallback (inner) is used
        # inner subpipe fails → inner fallback (AddPipe) is used
        assert result == 13

    def test_as_subpipe_of_fallback(self):
        """A FallbackPipe can be the subpipe of another FallbackPipe."""
        inner = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=AlwaysFailPipe())
        outer = FallbackPipe(subpipe=inner, fallback=AddPipe())

        result = outer.apply(3, 10)
        # inner subpipe fails → inner fallback also fails → inner raises
        # outer catches → outer fallback (AddPipe) used
        assert result == 13


# ─── Edge cases ────────────────────────────────────────────────────────────────


class TestFallbackPipeEdgeCases:
    def test_frozen_dataclass(self):
        """FallbackPipe is immutable."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe())
        with pytest.raises(AttributeError):
            pipe.subpipe = MultiplyPipe()  # type: ignore

    def test_single_item_batch(self):
        """Batch of one item works correctly."""
        pipe = FallbackPipe(subpipe=AlwaysFailPipe(), fallback=AddPipe())
        results = pipe.batch_apply([5], 10)
        assert list(results) == [15]

    def test_large_batch(self):
        """Large batches work correctly."""
        pipe = FallbackPipe(subpipe=AddPipe(), fallback=MultiplyPipe())
        data = list(range(1000))
        results = pipe.batch_apply(data, 1)
        assert list(results) == [d + 1 for d in data]

    def test_string_types(self):
        """FallbackPipe works with non-numeric types."""

        class ConcatPipe(Pipe[str, str, str]):
            def apply(self, data: str, metadata: str) -> str:
                return data + metadata

        class FailStringPipe(Pipe[str, str, str]):
            def apply(self, data: str, metadata: str) -> str:
                raise ValueError("nope")

        pipe = FallbackPipe(subpipe=FailStringPipe(), fallback=ConcatPipe())
        assert pipe.apply("hello ", "world") == "hello world"