import pytest

from kpipeline.async_pipeline import AsyncConditionalPipe
from test.async_pipeline.common import AsyncMultiplyPipe


@pytest.mark.asyncio
async def test_async_conditional_pipe():
    condition = lambda d, m: d % 2 == 0
    subpipe = AsyncMultiplyPipe()
    cond_pipe = AsyncConditionalPipe(condition, subpipe)
    
    assert await cond_pipe.apply(10, {}) == 20
    assert await cond_pipe.apply(11, {}) == 11
    assert await cond_pipe.async_batch_apply([10, 11], {}) == [20, 11]
