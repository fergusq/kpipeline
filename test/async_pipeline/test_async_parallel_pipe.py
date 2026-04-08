import pytest

from kpipeline.async_pipeline import AsyncParallelPipe
from test.async_pipeline.common import AsyncMultiplyPipe, AsyncSimplePipe


@pytest.mark.asyncio
async def test_async_parallel_pipe():
    p1 = AsyncSimplePipe()
    p2 = AsyncMultiplyPipe()
    combine = lambda results, m: sum(results)
    
    parallel = AsyncParallelPipe([p1, p2], combine)
    
    assert await parallel.apply(10, {"offset": 5}) == 35
    assert await parallel.async_batch_apply([10, 20], {"offset": 5}) == [35, 65]
