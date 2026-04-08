import pytest

from test.async_pipeline.common import AsyncMultiplyPipe, AsyncSimplePipe


@pytest.mark.asyncio
async def test_async_chain_pipe():
    p1 = AsyncSimplePipe()
    p2 = AsyncMultiplyPipe()
    
    # Using async_chain method
    chain1 = p1.async_chain(p2)
    assert await chain1.apply(10, {"offset": 5}) == 30 # (10+5)*2
    
    # Using | operator
    chain2 = p1 | p2
    assert await chain2.apply(10, {"offset": 5}) == 30
    
    assert await chain2.async_batch_apply([10, 20], {"offset": 5}) == [30, 50]
