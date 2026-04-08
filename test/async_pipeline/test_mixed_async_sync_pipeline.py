import pytest

from kpipeline.async_pipeline import AsyncChainPipe
from test.async_pipeline.common import AsyncMultiplyPipe, AsyncSimplePipe
from test.pipeline.common import MultiplyPipe, SimplePipe


@pytest.mark.asyncio
async def test_mixed_sync_async_chain():
    # Sync pipe followed by Async pipe
    p_sync = SimplePipe()
    p_async = AsyncMultiplyPipe()
    chain = AsyncChainPipe(p_sync, p_async)
    
    assert await chain.apply(10, {"offset": 5}) == 30
    
    # Async pipe followed by Sync pipe
    p_async2 = AsyncSimplePipe()
    p_sync2 = MultiplyPipe()
    chain2 = AsyncChainPipe(p_async2, p_sync2)
    
    assert await chain2.apply(10, {"offset": 5}) == 30
