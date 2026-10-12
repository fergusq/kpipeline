import asyncio
import time
from collections.abc import Sequence

import pytest

from kpipeline.async_pipeline import (
    AsyncIdentityPipe,
    AsyncMapPipe,
    AsyncPipe,
    AsyncRetryPipe,
    AsyncWrapperPipe,
)
from test.async_pipeline.common import (
    AsyncMultiplyPipe,
    AsyncSimplePipe,
    AsyncToStringPipe,
    MultiplyPipe,
    SimplePipe,
)


class TrackingAsyncPipe(AsyncPipe[int, int, dict]):
    """Records the inputs it receives through apply and async_batch_apply."""

    apply_calls: list[int]
    batch_calls: list[list[int]]

    def __init__(self) -> None:
        self.apply_calls = []
        self.batch_calls = []

    async def apply(self, data: int, metadata: dict) -> int:
        self.apply_calls.append(data)
        return data + metadata.get("offset", 0)

    async def async_batch_apply(self, data: Sequence[int], metadata: dict) -> list[int]:
        self.batch_calls.append(list(data))
        return [d + metadata.get("offset", 0) for d in data]


class AsyncAddOffsetPipe(AsyncPipe[int, int, dict]):
    """Adds the offset given in metadata to the input."""

    async def apply(self, data: int, metadata: dict) -> int:
        return data + metadata.get("offset", 0)


class AsyncTupleJoinPipe(AsyncPipe[tuple[int, str], str, dict]):
    """Joins the original input and the stringified input."""

    async def apply(self, data: tuple[int, str], metadata: dict) -> str:
        return f"{data[0]}->{data[1]}"


class SlowAsyncPipe(AsyncPipe[int, int, dict]):
    """Sleeps before doubling the input."""

    delay = 0.05

    async def apply(self, data: int, metadata: dict) -> int:
        await asyncio.sleep(self.delay)
        return data * 2


class FailingAsyncPipe(AsyncPipe[int, int, dict]):
    """Always fails."""

    async def apply(self, data: int, metadata: dict) -> int:
        raise ValueError(f"failed on {data}")


async def async_extract_value(data: dict, metadata: dict) -> int:
    return data["value"]


async def async_add_original_and_inner(pair: tuple[dict, int], metadata: dict) -> int:
    return pair[0]["value"] + pair[1]


def add_integers(pair: tuple[int, int], metadata: dict) -> int:
    return pair[0] + pair[1]


def count_outputs(pair: tuple[int, Sequence[int]], metadata: dict) -> int:
    return len(pair[1])


@pytest.mark.asyncio
async def test_apply_with_async_callables_and_async_subpipe():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, async_add_original_and_inner)
    assert await pipe.apply({"value": 10}, {}) == 30


@pytest.mark.asyncio
async def test_apply_with_sync_callables_and_sync_subpipe():
    pipe = AsyncWrapperPipe(MultiplyPipe(), lambda d, m: d, add_integers)
    assert await pipe.apply(3, {}) == 9


@pytest.mark.asyncio
async def test_apply_mixes_sync_and_async_parts():
    pipe = AsyncWrapperPipe(AsyncSimplePipe(), lambda d, m: d * 2, add_integers)
    assert await pipe.apply(3, {"offset": 5}) == 14


@pytest.mark.asyncio
async def test_apply_with_extract_pipe_and_wrap_pipe():
    pipe = AsyncWrapperPipe(AsyncToStringPipe(), AsyncAddOffsetPipe(), AsyncTupleJoinPipe())
    assert await pipe.apply(10, {"offset": 5}) == "10->15"


@pytest.mark.asyncio
async def test_wrap_receives_original_input_and_inner_output():
    seen: list[tuple[int, int]] = []

    async def wrap(pair: tuple[int, int], metadata: dict) -> int:
        seen.append(pair)
        return pair[1]

    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), lambda d, m: d * 3, wrap)
    assert await pipe.apply(2, {}) == 12
    assert seen == [(2, 12)]


@pytest.mark.asyncio
async def test_extract_and_wrap_receive_metadata():
    received: list[tuple[str, dict]] = []

    async def extract(data: int, metadata: dict) -> int:
        received.append(("extract", metadata))
        return data

    async def wrap(pair: tuple[int, int], metadata: dict) -> int:
        received.append(("wrap", metadata))
        return add_integers(pair, metadata) + metadata["bonus"]

    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), extract, wrap)
    assert await pipe.apply(3, {"bonus": 1}) == 10
    assert received == [("extract", {"bonus": 1}), ("wrap", {"bonus": 1})]


@pytest.mark.asyncio
async def test_extract_pipe_is_called_with_apply():
    extract = TrackingAsyncPipe()
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), extract, add_integers)

    assert await pipe.apply(1, {}) == 3
    assert extract.apply_calls == [1]
    assert extract.batch_calls == []


@pytest.mark.asyncio
async def test_subpipe_failure_propagates():
    pipe = AsyncWrapperPipe(FailingAsyncPipe(), async_extract_value, async_add_original_and_inner)
    with pytest.raises(ValueError, match="failed on 10"):
        await pipe.apply({"value": 10}, {})


@pytest.mark.asyncio
async def test_extract_failure_propagates():
    async def extract(data: int, metadata: dict) -> int:
        raise KeyError("no key")

    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), extract, add_integers)
    with pytest.raises(KeyError):
        await pipe.apply(1, {})


@pytest.mark.asyncio
async def test_wrap_failure_propagates():
    async def wrap(pair: tuple[int, int], metadata: dict) -> int:
        raise KeyError("no key")

    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), lambda d, m: d, wrap)
    with pytest.raises(KeyError):
        await pipe.apply(1, {})


@pytest.mark.asyncio
async def test_async_batch_apply_uses_subpipe_async_batch_apply():
    subpipe = TrackingAsyncPipe()
    pipe = AsyncWrapperPipe(subpipe, lambda d, m: d * 2, add_integers)

    assert await pipe.async_batch_apply([1, 2, 3], {"offset": 10}) == [13, 16, 19]
    assert subpipe.batch_calls == [[2, 4, 6]]
    assert subpipe.apply_calls == []


@pytest.mark.asyncio
async def test_async_batch_apply_with_sync_subpipe():
    pipe = AsyncWrapperPipe(SimplePipe(), lambda d, m: d * 2, add_integers)
    assert await pipe.async_batch_apply([1, 2], {"offset": 5}) == [8, 11]


@pytest.mark.asyncio
async def test_async_batch_apply_with_wrapped_map_pipe():
    pipe = AsyncWrapperPipe(AsyncMapPipe(AsyncMultiplyPipe()), lambda d, m: list(range(d)), count_outputs)
    assert await pipe.async_batch_apply([1, 3], {}) == [1, 3]
    assert await pipe.apply(3, {}) == 3


@pytest.mark.asyncio
async def test_batch_apply_returns_awaitables():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, async_add_original_and_inner)
    assert list(await asyncio.gather(*pipe.batch_apply([{"value": 1}, {"value": 2}], {}))) == [3, 6]


@pytest.mark.asyncio
async def test_async_batch_apply_with_empty_batch():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, async_add_original_and_inner)
    assert await pipe.async_batch_apply([], {}) == []


@pytest.mark.asyncio
async def test_extract_callables_run_concurrently():
    async def extract(data: int, metadata: dict) -> int:
        await asyncio.sleep(0.05)
        return data

    pipe = AsyncWrapperPipe(AsyncIdentityPipe(), extract, lambda pair, m: pair[1])

    start = time.monotonic()
    results = await pipe.async_batch_apply([1, 2, 3, 4, 5], {})
    elapsed = time.monotonic() - start

    assert list(results) == [1, 2, 3, 4, 5]
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_wrap_callables_run_concurrently():
    async def wrap(pair: tuple[int, int], metadata: dict) -> int:
        await asyncio.sleep(0.05)
        return pair[1]

    pipe = AsyncWrapperPipe(AsyncIdentityPipe(), lambda d, m: d, wrap)

    start = time.monotonic()
    results = await pipe.async_batch_apply([1, 2, 3, 4, 5], {})
    elapsed = time.monotonic() - start

    assert list(results) == [1, 2, 3, 4, 5]
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_subpipe_batch_runs_concurrently():
    pipe = AsyncWrapperPipe(SlowAsyncPipe(), lambda d, m: d, lambda pair, m: pair[1])

    start = time.monotonic()
    results = await pipe.async_batch_apply([1, 2, 3, 4, 5], {})
    elapsed = time.monotonic() - start

    assert list(results) == [2, 4, 6, 8, 10]
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_chaining_with_async_and_sync_pipes():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, async_add_original_and_inner)
    assert await (pipe | AsyncMultiplyPipe()).apply({"value": 10}, {}) == 60
    assert await (pipe | MultiplyPipe()).apply({"value": 10}, {}) == 60


@pytest.mark.asyncio
async def test_wrapper_of_wrapper():
    inner = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, async_add_original_and_inner)
    outer = AsyncWrapperPipe(inner, lambda d, m: {"value": d}, lambda pair, m: pair[1])
    assert await outer.apply(5, {}) == 15


@pytest.mark.asyncio
async def test_wrapper_inside_map_pipe():
    pipe = AsyncMapPipe(AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, async_add_original_and_inner))
    assert await pipe.apply([{"value": 1}, {"value": 2}], {}) == [3, 6]


def test_is_wrapper_and_subgraph():
    subpipe = AsyncAddOffsetPipe()
    pipe = AsyncWrapperPipe(subpipe, async_extract_value, add_integers)
    assert pipe.is_wrapper() is True
    assert pipe.get_subgraph() == subpipe.to_graph()


def test_to_node_title_and_shape():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, add_integers, description="My wrapper")
    node = pipe.to_node()
    assert node.title == "My wrapper"
    assert node.shape == "combine"
    assert node.subgraph is None


def test_to_node_has_wrap_pipe_as_subgraph():
    wrap = AsyncTupleJoinPipe()
    pipe = AsyncWrapperPipe(AsyncToStringPipe(), AsyncAddOffsetPipe(), wrap)
    assert pipe.to_node().subgraph == wrap.to_graph()


def test_to_graph_is_valid():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, add_integers)
    assert pipe.to_graph().is_valid()


def test_to_graph_is_valid_with_async_extract_pipe_and_complex_subpipe():
    subpipe = AsyncRetryPipe(FailingAsyncPipe(), retries=1, exceptions=(ValueError,))
    pipe = AsyncWrapperPipe(subpipe, AsyncAddOffsetPipe(), lambda pair, m: pair[1])
    assert pipe.to_graph().is_valid()


def test_to_graph_inputs_and_outputs():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, add_integers)
    graph = pipe.to_graph()
    combine_id = str(id(pipe))

    assert graph.outputs == (combine_id,)
    assert combine_id in graph.inputs
    assert f"{combine_id}_extract" in graph.inputs


def test_to_graph_connects_subpipe_outputs_to_combine_node():
    subpipe = AsyncMultiplyPipe()
    pipe = AsyncWrapperPipe(subpipe, async_extract_value, add_integers)
    graph = pipe.to_graph()

    subpipe_ids = {node.id for node in subpipe.to_graph().nodes}
    connections = [c for c in graph.connections if c.to_node == str(id(pipe))]
    assert {c.from_node for c in connections} == subpipe_ids


def test_to_graph_uses_extract_pipe_graph():
    extract = AsyncAddOffsetPipe()
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), extract, add_integers)
    graph = pipe.to_graph()
    extract_ids = {node.id for node in extract.to_graph().nodes}

    assert extract_ids <= {node.id for node in graph.nodes}
    assert not any(node.id.endswith("_extract") for node in graph.nodes)
    assert graph.inputs == extract.to_graph().inputs + (str(id(pipe)),)


def test_to_graph_uses_extract_description_as_node_title():
    pipe = AsyncWrapperPipe(
        AsyncMultiplyPipe(),
        async_extract_value,
        add_integers,
        extract_description="Pick value",
    )
    extract_node = pipe.to_graph().get_node(f"{id(pipe)}_extract")
    assert extract_node is not None
    assert extract_node.title == "Pick value"


def test_to_graph_omits_extract_node_when_description_is_none():
    subpipe = AsyncMultiplyPipe()
    pipe = AsyncWrapperPipe(subpipe, async_extract_value, add_integers, extract_description=None)
    graph = pipe.to_graph()

    assert {node.id for node in graph.nodes} == {str(id(pipe)), str(id(subpipe))}
    assert graph.inputs == subpipe.to_graph().inputs + (str(id(pipe)),)


def test_to_dot_renders_wrapper():
    pipe = AsyncWrapperPipe(
        AsyncToStringPipe(),
        AsyncAddOffsetPipe(),
        AsyncTupleJoinPipe(),
        description="Wrapped pipe",
    )
    dot = pipe.to_graph().to_dot()
    assert 'label="Wrapped pipe"' in dot
    assert "AsyncAddOffsetPipe" in dot


def test_to_dot_renders_extract_node():
    pipe = AsyncWrapperPipe(
        AsyncMultiplyPipe(),
        async_extract_value,
        add_integers,
        extract_description="Pick value",
    )
    assert "Pick value" in pipe.to_graph().to_dot()


def test_is_immutable():
    pipe = AsyncWrapperPipe(AsyncMultiplyPipe(), async_extract_value, add_integers)
    with pytest.raises(AttributeError):
        pipe.subpipe = SimplePipe()  # type: ignore
