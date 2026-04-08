import pytest

from kpipeline.async_pipeline import AsyncMapPipe
from test.async_pipeline.common import AsyncSimplePipe


@pytest.mark.asyncio
async def test_async_map_pipe():
    subpipe = AsyncSimplePipe()
    map_pipe = AsyncMapPipe(subpipe)
    
    assert await map_pipe.apply([10, 20], {"offset": 5}) == [15, 25]
