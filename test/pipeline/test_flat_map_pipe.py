from kpipeline.pipeline import FlatMapPipe, Pipe
from kpipeline.graph import Graph


class DuplicatePipe(Pipe[int, list[int], dict]):
    def apply(self, data: int, metadata: dict) -> list[int]:
        return [data, data]


def test_flat_map_pipe():
    subpipe = DuplicatePipe()
    flat_map_pipe = FlatMapPipe(subpipe)

    assert flat_map_pipe.apply([1, 2, 3], {"offset": 0}) == [1, 1, 2, 2, 3, 3]


def test_flat_map_pipe_batch_apply():
    subpipe = DuplicatePipe()
    flat_map_pipe = FlatMapPipe(subpipe)

    result = flat_map_pipe.batch_apply([[1, 2], [3]], {"offset": 0})
    assert result == [[1, 1, 2, 2], [3, 3]]


def test_flat_map_pipe_get_subgraph():
    subpipe = DuplicatePipe()
    flat_map_pipe = FlatMapPipe(subpipe)

    assert isinstance(flat_map_pipe.get_subgraph(), Graph)


def test_flat_map_pipe_to_node():
    subpipe = DuplicatePipe()
    flat_map_pipe = FlatMapPipe(subpipe)

    node = flat_map_pipe.to_node()
    assert node.title == "FlatMap"
