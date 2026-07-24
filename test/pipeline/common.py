from kpipeline.pipeline import Pipe


class SimplePipe(Pipe[int, int, dict]):
    def apply(self, data: int, metadata: dict) -> int:
        return data + metadata.get("offset", 0)

class MultiplyPipe(Pipe[int, int, dict]):
    def apply(self, data: int, metadata: dict) -> int:
        return data * 2

class ToStringPipe(Pipe[int, str, dict]):
    def apply(self, data: int, metadata: dict) -> str:
        return str(data)
