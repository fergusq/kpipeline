import pytest

from test.async_pipeline.common import AsyncSimplePipe

@pytest.mark.asyncio
async def test_async_simple_pipe():
    pipe = AsyncSimplePipe()
    assert await pipe.apply(10, {"offset": 5}) == 15
    assert await pipe.async_batch_apply([10, 20], {"offset": 5}) == [15, 25]
