import builtins
import types
import pytest

from kpipeline.graph import (
    Graph,
    GraphNode,
    GraphConnection,
)

from kpipeline.pipeline import (
    Pipe,
    ChainPipe,
    BranchPipe,
    ConditionalPipe,
    SelectPipe,
    ParallelPipe,
    MetadataWrapperPipe,
    MapPipe,
    FilterPipe,
    RetryPipe,
)

# ----------------------------------------------------------------------
# Helper concrete Pipe implementations (the abstract Pipe class only
# defines the interface – we need real behaviour for the tests)
# ----------------------------------------------------------------------


class AddPipe(Pipe[int, int, None]):
    """Adds 1 to the incoming integer."""

    def apply(self, input: int, metadata: None) -> int:  # type: ignore[override]
        return input + 1


class MulPipe(Pipe[int, int, None]):
    """Multiplies the incoming integer by 2."""

    def apply(self, input: int, metadata: None) -> int:  # type: ignore[override]
        return input * 2


# ----------------------------------------------------------------------
# Pipe chaining (`Pipe.chain` / `Pipe.__or__`) and concrete pipe behaviour
# ----------------------------------------------------------------------


def test_pipe_chain_operator_produces_chainpipe_and_composes_results():
    add = AddPipe()
    mul = MulPipe()

    # Using the explicit method
    chain1 = add.chain(mul)
    assert isinstance(chain1, ChainPipe)

    # Using the `|` operator (syntactic sugar)
    chain2 = add | mul
    assert isinstance(chain2, ChainPipe)

    # Both chains must behave the same
    for chain in (chain1, chain2):
        # (x + 1) * 2
        assert chain.apply(3, None) == (3 + 1) * 2
        # Graph representation must be a valid graph containing two nodes
        g = chain.to_graph()
        assert isinstance(g, Graph)
        assert len(g.nodes) == 2
        assert g.is_valid()


# ----------------------------------------------------------------------
# BranchPipe – conditional execution of one of two sub‑pipes
# ----------------------------------------------------------------------


def test_branch_pipe_applies_correct_branch():
    condition = lambda x, _: x > 0
    branch = BranchPipe(condition=condition, then_pipe=AddPipe(), else_pipe=MulPipe(), description="pos?")

    # Positive numbers go through `then_pipe` (AddPipe)
    assert branch.apply(5, None) == 6
    # Zero / negative numbers go through `else_pipe` (MulPipe)
    assert branch.apply(-2, None) == -4
    assert branch.apply(0, None) == 0

    # Graph must be valid (the internal implementation builds a condition node and links to the two sub‑graphs)
    g = branch.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()
    # The condition node should be present exactly once
    condition_nodes = [n for n in g.nodes if n.title == "pos?" or n.shape == "condition"]
    assert len(condition_nodes) == 1


# ----------------------------------------------------------------------
# ConditionalPipe – apply a pipe only when a predicate holds, otherwise identity
# ----------------------------------------------------------------------


def test_conditional_pipe_behaviour():
    cond = ConditionalPipe(condition=lambda x, _: x % 2 == 0,
                           subpipe=MulPipe(),
                           description="even?")

    # Even numbers are multiplied
    assert cond.apply(4, None) == 8
    # Odd numbers are returned unchanged
    assert cond.apply(5, None) == 5

    g = cond.to_graph()
    assert isinstance(g, Graph)
    # The graph contains the condition node *and* the sub‑pipe node
    assert any(n.title == "even?" for n in g.nodes)
    assert g.is_valid()


# ----------------------------------------------------------------------
# SelectPipe – route to one of many sub‑pipes based on a selector key
# ----------------------------------------------------------------------


def test_select_pipe_routing_and_graph():
    subpipes = {
        0: AddPipe(),
        1: MulPipe(),
    }
    selector = lambda x, _: x % 3
    select = SelectPipe(key=selector,
                        subpipes=subpipes,
                        otherwise_pipe=AddPipe(),
                        description="mod‑3")

    # 0 -> key 0 -> AddPipe
    assert select.apply(6, None) == 7
    # 1 -> key 1 -> MulPipe
    assert select.apply(4, None) == 8
    # 2 -> key 2 not in mapping -> otherwise (AddPipe)
    assert select.apply(2, None) == 3

    g = select.to_graph()
    assert isinstance(g, Graph)
    # The graph must contain at least one condition node and all sub‑pipe nodes
    assert any(n.title == "mod‑3" for n in g.nodes)
    assert g.is_valid()


# ----------------------------------------------------------------------
# ParallelPipe – run several pipes on the same input and combine the results
# ----------------------------------------------------------------------


def test_parallel_pipe_combines_results():
    # combine function just sums the intermediate results
    combiner = lambda results, _: sum(results)

    parallel = ParallelPipe(
        subpipes=[AddPipe(), MulPipe()],  # (x+1) and (x*2)
        combine=combiner,
        description="sum‑parallel",
    )

    # (3+1) + (3*2) = 4 + 6 = 10
    assert parallel.apply(3, None) == 10

    g = parallel.to_graph()
    assert isinstance(g, Graph)
    # The combine node must be present once
    combine_nodes = [n for n in g.nodes if n.title == "sum‑parallel" or n.shape == "combine"]
    assert len(combine_nodes) == 1
    assert g.is_valid()


# ----------------------------------------------------------------------
# MetadataWrapperPipe – transform outer metadata before delegating to a sub‑pipe
# ----------------------------------------------------------------------


def test_metadata_wrapper_pipe_transforms_metadata():
    # The inner pipe just multiplies the input
    inner = MulPipe()

    # Outer metadata is a dict; we extract the value under key "factor"
    wrapper = MetadataWrapperPipe(
        to_inner=lambda outer: outer["factor"],
        subpipe=inner,
        description="unwrap‑factor",
    )

    # Provide metadata where the inner factor is 2 (the MulPipe ignores its own metadata)
    outer_meta = {"factor": None}  # the value is not used by MulPipe, just testing the plumbing
    assert wrapper.apply(5, outer_meta) == 10

    g = wrapper.to_graph()
    assert isinstance(g, Graph)
    # The wrapper node should appear in the graph
    assert any(n.title == "unwrap‑factor" for n in g.nodes)
    assert g.is_valid()


# ----------------------------------------------------------------------
# Edge‑case: chaining more than two pipes works recursively
# ----------------------------------------------------------------------


def test_multiple_chain_composition():
    chain = AddPipe().chain(MulPipe()).chain(AddPipe())  # ((x+1)*2)+1
    assert isinstance(chain, ChainPipe)
    assert chain.apply(2, None) == ((2 + 1) * 2) + 1  # 7

    # The graph should contain three nodes (one per pipe)
    g = chain.to_graph()
    assert len(g.nodes) == 3
    assert g.is_valid()


# ----------------------------------------------------------------------
# Ensure that the abstract base class `Pipe` cannot be instantiated directly
# ----------------------------------------------------------------------


def test_pipe_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Pipe()  # type: ignore


# ----------------------------------------------------------------------
# Helper concrete pipes used by the tests
# ----------------------------------------------------------------------


class FailingPipe(Pipe[int, int, None]):  # type: ignore[misc]
    """Pipe that raises on the first call, then succeeds."""
    _called: bool = False

    def apply(self, input: int, metadata: None) -> int:  # noqa: D401
        if not getattr(self, "_called", False):
            self.__dict__["_called"] = True
            raise RuntimeError("first failure")
        return input * 2

    def to_graph(self) -> Graph:  # pragma: no cover
        node = self.to_node()
        return Graph(nodes=(node,), connections=(), inputs=(node.id,), outputs=(node.id,))


# ----------------------------------------------------------------------
# MapPipe
# ----------------------------------------------------------------------

def test_map_pipe():
    pipe = MapPipe(MulPipe())
    assert pipe.apply([1, 2, 3], None) == [2, 4, 6]

    g = pipe.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()


# ----------------------------------------------------------------------
# FilterPipe
# ----------------------------------------------------------------------

def test_filter_pipe():
    pred = lambda x, m: x % 2 == 0
    pipe = FilterPipe(predicate=pred)
    assert pipe.apply([1, 2, 3, 4], None) == [2, 4]

    g = pipe.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()

# ----------------------------------------------------------------------
# RetryPipe
# ----------------------------------------------------------------------

def test_retry_pipe_successful_retry():
    pipe = RetryPipe(subpipe=FailingPipe(), retries=2, exceptions=RuntimeError)
    # First call fails, second succeeds -> result should be 5*2 = 10
    assert pipe.apply(5, None) == 10

    g = pipe.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()


def test_retry_pipe_exhausts():
    class AlwaysFail(Pipe[int, int, None]):  # type: ignore[misc]
        def apply(self, input: int, metadata: None) -> int:  # noqa: D401
            raise RuntimeError("always fail")

        def to_graph(self) -> Graph:  # pragma: no cover
            node = self.to_node()
            return Graph(nodes=(node,), connections=(), inputs=(node.id,), outputs=(node.id,))

    pipe = RetryPipe(subpipe=AlwaysFail(), retries=1, exceptions=RuntimeError)
    with pytest.raises(RuntimeError):
        pipe.apply(1, None)


