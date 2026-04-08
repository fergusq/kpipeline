import pytest

from kpipeline.async_pipeline import AsyncIdentityPipe

@pytest.mark.asyncio
async def test_async_identity_pipe():
    pipe = AsyncIdentityPipe[int, dict]()
    assert await pipe.apply(10, {}) == 10
    assert await pipe.async_batch_apply([10, 20], {}) == [10, 20]
