import pytest

from kpipeline.async_pipeline import AsyncMetadataWrapperPipe
from test.async_pipeline.common import AsyncSimplePipe


@pytest.mark.asyncio
async def test_async_metadata_wrapper_pipe():
    async def to_inner(m): return {"offset": m["value"] * 2}
    subpipe = AsyncSimplePipe()
    wrapper = AsyncMetadataWrapperPipe(to_inner, subpipe)
    
    assert await wrapper.apply(10, {"value": 5}) == 20
