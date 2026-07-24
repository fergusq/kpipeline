import pytest

from kpipeline.async_pipeline import AsyncParallel2Pipe
from test.async_pipeline.common import AsyncSimplePipe, AsyncToStringPipe


@pytest.mark.asyncio
async def test_async_parallel2_pipe():
    p1 = AsyncSimplePipe()
    p2 = AsyncToStringPipe()
    combine = lambda results, m: f"{repr(results[0])} {repr(results[1])}"
    
    parallel = AsyncParallel2Pipe(p1, p2, combine)
    
    assert await parallel.apply(10, {"offset": 5}) == "15 '10'"
    assert await parallel.async_batch_apply([10, 20], {"offset": 5}) == ["15 '10'", "25 '20'"]
