import asyncio
import pytest

from kpipeline.async_pipeline import AsyncFlatMapPipe, AsyncPipe


class AsyncDuplicatePipe(AsyncPipe[int, list[int], dict]):
    async def apply(self, data: int, metadata: dict) -> list[int]:
        await asyncio.sleep(0.01)
        return [data, data]


@pytest.mark.asyncio
async def test_async_flat_map_pipe():
    subpipe = AsyncDuplicatePipe()
    flat_map_pipe = AsyncFlatMapPipe(subpipe)

    assert await flat_map_pipe.apply([1, 2, 3], {"offset": 0}) == [1, 1, 2, 2, 3, 3]


@pytest.mark.asyncio
async def test_async_flat_map_pipe_batch_apply():
    subpipe = AsyncDuplicatePipe()
    flat_map_pipe = AsyncFlatMapPipe(subpipe)

    result = await flat_map_pipe.async_batch_apply([[1, 2], [3]], {"offset": 0})
    assert result == [[1, 1, 2, 2], [3, 3]]


@pytest.mark.asyncio
async def test_async_flat_map_pipe_get_subgraph():
    from kpipeline.graph import Graph

    subpipe = AsyncDuplicatePipe()
    flat_map_pipe = AsyncFlatMapPipe(subpipe)

    assert isinstance(flat_map_pipe.get_subgraph(), Graph)


@pytest.mark.asyncio
async def test_async_flat_map_pipe_to_node():
    subpipe = AsyncDuplicatePipe()
    flat_map_pipe = AsyncFlatMapPipe(subpipe)

    node = flat_map_pipe.to_node()
    assert node.title == "FlatMap"
