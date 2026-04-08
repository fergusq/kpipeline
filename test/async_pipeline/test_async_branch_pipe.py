import pytest

from kpipeline.async_pipeline import AsyncBranchPipe
from test.async_pipeline.common import AsyncMultiplyPipe, AsyncSimplePipe


@pytest.mark.asyncio
async def test_async_branch_pipe():
    async def condition(d, m): return d > 15
    p_then = AsyncMultiplyPipe()
    p_else = AsyncSimplePipe()
    branch = AsyncBranchPipe(condition, p_then, p_else)
    
    assert await branch.apply(20, {"offset": 5}) == 40
    assert await branch.apply(10, {"offset": 5}) == 15
    assert await branch.async_batch_apply([20, 10], {"offset": 5}) == [40, 15]
