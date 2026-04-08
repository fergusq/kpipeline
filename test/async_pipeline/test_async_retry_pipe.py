import pytest

from kpipeline.async_pipeline import AsyncPipe, AsyncRetryPipe


@pytest.mark.asyncio
async def test_async_retry_pipe():
    class AsyncFailingPipe(AsyncPipe[int, int, dict]):
        def __init__(self):
            self.count = 0
        async def apply(self, data, metadata):
            self.count += 1
            if self.count < 3:
                raise ValueError("Fail")
            return data
            
    failing = AsyncFailingPipe()
    retry = AsyncRetryPipe(failing, retries=3, exceptions=ValueError)
    
    assert await retry.apply(10, {}) == 10
    assert failing.count == 3

@pytest.mark.asyncio
async def test_async_retry_pipe_exhausted():
    class AsyncAlwaysFailingPipe(AsyncPipe[int, int, dict]):
        async def apply(self, data, metadata):
            raise ValueError("Fail")
            
    retry = AsyncRetryPipe(AsyncAlwaysFailingPipe(), retries=2, exceptions=ValueError)
    
    with pytest.raises(ValueError):
        await retry.apply(10, {})