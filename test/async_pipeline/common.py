import asyncio

from kpipeline.async_pipeline import AsyncPipe
from kpipeline.pipeline import Pipe


class SimplePipe(Pipe[int, int, dict]):
    def apply(self, data: int, metadata: dict) -> int:
        return data + metadata.get("offset", 0)

class MultiplyPipe(Pipe[int, int, dict]):
    def apply(self, data: int, metadata: dict) -> int:
        return data * 2

class AsyncSimplePipe(AsyncPipe[int, int, dict]):
    async def apply(self, data: int, metadata: dict) -> int:
        await asyncio.sleep(0.01)
        return data + metadata.get("offset", 0)

class AsyncMultiplyPipe(AsyncPipe[int, int, dict]):
    async def apply(self, data: int, metadata: dict) -> int:
        await asyncio.sleep(0.01)
        return data * 2

class AsyncToStringPipe(Pipe[int, str, dict]):
    def apply(self, data: int, metadata: dict) -> str:
        return str(data)
