from collections.abc import Sequence

import pytest

from kpipeline.pipeline import MapPipe, Pipe, RetryPipe, WrapperPipe
from test.pipeline.common import MultiplyPipe, SimplePipe, ToStringPipe


class AddOffsetPipe(Pipe[int, int, dict]):
    """Adds the offset given in metadata to the input."""

    def apply(self, data: int, metadata: dict) -> int:
        return data + metadata.get("offset", 0)


class TrackingPipe(Pipe[int, int, dict]):
    """Records the inputs it receives through apply and batch_apply."""

    apply_calls: list[int]
    batch_calls: list[list[int]]

    def __init__(self) -> None:
        self.apply_calls = []
        self.batch_calls = []

    def apply(self, data: int, metadata: dict) -> int:
        self.apply_calls.append(data)
        return data + metadata.get("offset", 0)

    def batch_apply(self, data: Sequence[int], metadata: dict) -> list[int]:
        self.batch_calls.append(list(data))
        return [d + metadata.get("offset", 0) for d in data]


class FailOnOddPipe(Pipe[int, int, dict]):
    """Fails on odd inputs and doubles even inputs."""

    def apply(self, data: int, metadata: dict) -> int:
        if data % 2:
            raise ValueError(f"odd input {data}")
        return data * 2


class TupleJoinPipe(Pipe[tuple[int, str], str, dict]):
    """Joins the original input and the stringified input."""

    def apply(self, data: tuple[int, str], metadata: dict) -> str:
        return f"{data[0]}->{data[1]}"


def extract_value(data: dict, metadata: dict) -> int:
    return data["value"]


def add_original_and_inner(pair: tuple[dict, int], metadata: dict) -> int:
    return pair[0]["value"] + pair[1]


def add_integers(pair: tuple[int, int], metadata: dict) -> int:
    return pair[0] + pair[1]


def count_outputs(pair: tuple[int, Sequence[int]], metadata: dict) -> int:
    return len(pair[1])


def test_apply_with_callables():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_original_and_inner)
    assert pipe.apply({"value": 10}, {}) == 30


def test_apply_with_extract_pipe_and_wrap_pipe():
    pipe = WrapperPipe(subpipe=ToStringPipe(), extract=AddOffsetPipe(), wrap=TupleJoinPipe())
    assert pipe.apply(10, {"offset": 5}) == "10->15"


def test_wrap_receives_original_input_and_inner_output():
    seen: list[tuple[int, int]] = []

    def wrap(pair: tuple[int, int], metadata: dict) -> int:
        seen.append(pair)
        return pair[1]

    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=lambda d, m: d * 3, wrap=wrap)
    assert pipe.apply(2, {}) == 12
    assert seen == [(2, 12)]


def test_extract_and_wrap_receive_metadata():
    received: list[tuple[str, dict]] = []

    def extract(data: int, metadata: dict) -> int:
        received.append(("extract", metadata))
        return data

    def wrap(pair: tuple[int, int], metadata: dict) -> int:
        received.append(("wrap", metadata))
        return add_integers(pair, metadata) + metadata["bonus"]

    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract, wrap=wrap)
    assert pipe.apply(3, {"bonus": 1}) == 10
    assert received == [("extract", {"bonus": 1}), ("wrap", {"bonus": 1})]


def test_extract_pipe_is_called_with_apply():
    extract = TrackingPipe()
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract, wrap=add_integers)

    assert pipe.apply(1, {}) == 3
    assert extract.apply_calls == [1]
    assert extract.batch_calls == []


def test_subpipe_failure_propagates():
    pipe = WrapperPipe(subpipe=FailOnOddPipe(), extract=extract_value, wrap=add_original_and_inner)
    with pytest.raises(ValueError, match="odd input 3"):
        pipe.apply({"value": 3}, {})


def test_extract_failure_propagates():
    def extract(data: int, metadata: dict) -> int:
        raise KeyError("no key")

    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract, wrap=add_integers)
    with pytest.raises(KeyError):
        pipe.apply(1, {})


def test_batch_apply_uses_subpipe_batch_apply():
    subpipe = TrackingPipe()
    pipe = WrapperPipe(subpipe=subpipe, extract=lambda d, m: d * 2, wrap=add_integers)

    assert list(pipe.batch_apply([1, 2, 3], {"offset": 10})) == [13, 16, 19]
    assert subpipe.batch_calls == [[2, 4, 6]]
    assert subpipe.apply_calls == []


def test_batch_apply_with_wrapped_map_pipe():
    pipe = WrapperPipe(subpipe=MapPipe(MultiplyPipe()), extract=lambda d, m: list(range(d)), wrap=count_outputs)
    assert list(pipe.batch_apply([1, 3], {})) == [1, 3]
    assert pipe.apply(3, {}) == 3


def test_batch_apply_with_empty_batch():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_original_and_inner)
    assert list(pipe.batch_apply([], {})) == []


def test_chaining_with_other_pipes():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_original_and_inner) | MultiplyPipe()
    assert pipe.apply({"value": 10}, {}) == 60


def test_wrapper_of_wrapper():
    inner = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_original_and_inner)
    outer = WrapperPipe(subpipe=inner, extract=lambda d, m: {"value": d}, wrap=lambda pair, m: pair[1])
    assert outer.apply(5, {}) == 15


def test_is_wrapper_and_subgraph():
    subpipe = AddOffsetPipe()
    pipe = WrapperPipe(subpipe=subpipe, extract=extract_value, wrap=add_integers)
    assert pipe.is_wrapper() is True
    assert pipe.get_subgraph() == subpipe.to_graph()


def test_to_node_title_and_shape():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_integers, description="My wrapper")
    node = pipe.to_node()
    assert node.title == "My wrapper"
    assert node.shape == "combine"
    assert node.subgraph is None


def test_to_node_has_wrap_pipe_as_subgraph():
    wrap = TupleJoinPipe()
    pipe = WrapperPipe(subpipe=ToStringPipe(), extract=AddOffsetPipe(), wrap=wrap)
    assert pipe.to_node().subgraph == wrap.to_graph()


def test_to_graph_is_valid():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_integers)
    assert pipe.to_graph().is_valid()


def test_to_graph_is_valid_with_complex_subpipe():
    subpipe = RetryPipe(FailOnOddPipe(), retries=1, exceptions=(ValueError,))
    pipe = WrapperPipe(subpipe=subpipe, extract=extract_value, wrap=add_original_and_inner, extract_description=None)
    assert pipe.to_graph().is_valid()


def test_to_graph_inputs_and_outputs():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_integers)
    graph = pipe.to_graph()
    combine_id = str(id(pipe))

    assert graph.outputs == (combine_id,)
    assert combine_id in graph.inputs
    assert f"{combine_id}_extract" in graph.inputs


def test_to_graph_connects_subpipe_outputs_to_combine_node():
    subpipe = MultiplyPipe()
    pipe = WrapperPipe(subpipe=subpipe, extract=extract_value, wrap=add_integers)
    graph = pipe.to_graph()

    subpipe_ids = {node.id for node in subpipe.to_graph().nodes}
    connections = [c for c in graph.connections if c.to_node == str(id(pipe))]
    assert {c.from_node for c in connections} == subpipe_ids


def test_to_graph_uses_extract_pipe_graph():
    extract = AddOffsetPipe()
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract, wrap=add_integers)
    graph = pipe.to_graph()
    extract_ids = {node.id for node in extract.to_graph().nodes}

    assert extract_ids <= {node.id for node in graph.nodes}
    assert not any(node.id.endswith("_extract") for node in graph.nodes)
    assert graph.inputs == extract.to_graph().inputs + (str(id(pipe)),)


def test_to_graph_uses_extract_description_as_node_title():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_integers, extract_description="Pick value")
    graph = pipe.to_graph()
    extract_node = graph.get_node(f"{id(pipe)}_extract")
    assert extract_node is not None
    assert extract_node.title == "Pick value"


def test_to_graph_omits_extract_node_when_description_is_none():
    subpipe = MultiplyPipe()
    pipe = WrapperPipe(subpipe=subpipe, extract=extract_value, wrap=add_integers, extract_description=None)
    graph = pipe.to_graph()

    assert {node.id for node in graph.nodes} == {str(id(pipe)), str(id(subpipe))}
    assert graph.inputs == subpipe.to_graph().inputs + (str(id(pipe)),)


def test_to_dot_renders_wrapper():
    pipe = WrapperPipe(
        subpipe=ToStringPipe(),
        extract=AddOffsetPipe(),
        wrap=TupleJoinPipe(),
        description="Wrapped pipe",
    )
    dot = pipe.to_graph().to_dot()
    assert 'label="Wrapped pipe"' in dot
    assert "AddOffsetPipe" in dot


def test_to_dot_renders_extract_node():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_integers, extract_description="Pick value")
    dot = pipe.to_graph().to_dot()
    assert "Pick value" in dot


def test_is_immutable():
    pipe = WrapperPipe(subpipe=MultiplyPipe(), extract=extract_value, wrap=add_integers)
    with pytest.raises(AttributeError):
        pipe.subpipe = SimplePipe()  # type: ignore
