import pytest

from kpipeline.async_pipeline import AsyncIdentityPipe, AsyncSelectPipe
from test.async_pipeline.common import AsyncMultiplyPipe, AsyncSimplePipe


@pytest.mark.asyncio
async def test_async_select_pipe():
    async def get_key(d, m): return "even" if d % 2 == 0 else "odd"
    
    p_even = AsyncMultiplyPipe()
    p_odd = AsyncSimplePipe()
    p_otherwise = AsyncIdentityPipe[int, dict]()
    
    select = AsyncSelectPipe(get_key, {"even": p_even, "odd": p_odd}, p_otherwise)
    
    assert await select.apply(10, {"offset": 5}) == 20
    assert await select.apply(11, {"offset": 5}) == 16
    assert await select.async_batch_apply([10, 11], {"offset": 5}) == [20, 16]
