import builtins
import types
import pytest

from kpipe.graph import (
    Graph,
    GraphNode,
    GraphConnection,
)


# ----------------------------------------------------------------------
# Graph unit‑tests
# ----------------------------------------------------------------------


def test_graph_addition_and_validity():
    """`Graph.__or__` must concatenate the underlying tuples."""
    n1 = GraphNode(id="a", title="A")
    n2 = GraphNode(id="b", title="B")
    c1 = GraphConnection("a", "b", "link")

    g1 = Graph(nodes=(n1,), connections=(c1,), inputs=("a",), outputs=("b",))
    g2 = Graph(nodes=(n2,), connections=(), inputs=(), outputs=())

    g3 = g1 | g2

    assert isinstance(g3, Graph)
    assert g3.nodes == (n1, n2)
    assert g3.connections == (c1,)
    assert g3.inputs == ("a",)
    assert g3.outputs == ("b",)

    # The graph is valid because every reference points to an existing node.
    assert g3.is_valid()


def test_graph_rshift_creates_cross_connections():
    """`Graph.__rshift__` must connect every output of the left graph to every input of the right graph."""
    n_left = GraphNode(id="l", title="L")
    n_right = GraphNode(id="r", title="R")

    left = Graph(nodes=(n_left,), connections=(), inputs=("l",), outputs=("l",))
    right = Graph(nodes=(n_right,), connections=(), inputs=("r",), outputs=("r",))

    composed = left >> right

    # Two original nodes plus the automatically created connection
    assert composed.nodes == (n_left, n_right)
    assert len(composed.connections) == 1
    conn = composed.connections[0]
    assert conn.from_node == "l"
    assert conn.to_node == "r"
    assert conn.label == ""  # default label

    # Inputs come from the left side, outputs from the right side
    assert composed.inputs == ("l",)
    assert composed.outputs == ("r",)

    # The composed graph is still valid
    assert composed.is_valid()


def test_graph_is_invalid_when_reference_missing():
    n = GraphNode(id="x", title="X")
    g = Graph(nodes=(n,), connections=(GraphConnection("x", "y"),), inputs=("x",), outputs=("x",))
    assert not g.is_valid()  # "y" does not exist
