import pytest

from kpipeline.async_pipeline import AsyncIdentityPipe, AsyncMerge2Pipe, AsyncMerge3Pipe
from test.async_pipeline.common import AsyncMultiplyPipe, AsyncSimplePipe


@pytest.mark.asyncio
async def test_async_merge2_pipe():
    p1 = AsyncSimplePipe()
    p2 = AsyncMultiplyPipe()
    merge = AsyncMerge2Pipe(p1, p2, lambda a, b: a + b)
    
    assert await merge.apply(10, {"offset": 5}) == 35
    assert await merge.async_batch_apply([10, 20], {"offset": 5}) == [35, 65]

@pytest.mark.asyncio
async def test_async_merge3_pipe():
    p1 = AsyncSimplePipe()
    p2 = AsyncMultiplyPipe()
    p3 = AsyncIdentityPipe[int, dict]()
    merge = AsyncMerge3Pipe(p1, p2, p3, lambda a, b, c: a + b + c)
    
    assert await merge.apply(10, {"offset": 5}) == 45
