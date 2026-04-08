import pytest

from kpipeline.async_pipeline import AsyncFilterPipe


@pytest.mark.asyncio
async def test_async_filter_pipe():
    async def predicate(d, m): return d > 15
    filter_pipe = AsyncFilterPipe(predicate)
    
    assert await filter_pipe.apply([10, 20, 30], {}) == [20, 30]