from typing import NamedTuple, Optional

type GraphNodeId = str


class GraphNode(NamedTuple):
    id: GraphNodeId
    title: str
    type: Optional[type] = None
    shape: Optional[str] = None
    subgraph: "Optional[Graph]" = None


class GraphConnection(NamedTuple):
    from_node: GraphNodeId
    to_node: GraphNodeId
    label: str = ""


class Graph(NamedTuple):
    nodes: tuple[GraphNode, ...] = ()
    connections: tuple[GraphConnection, ...] = ()
    inputs: tuple[GraphNodeId, ...] = ()
    outputs: tuple[GraphNodeId, ...] = ()

    def __or__(self, other: "Graph") -> "Graph":
        return Graph(
            nodes=self.nodes + other.nodes,
            connections=self.connections + other.connections,
            inputs=self.inputs + other.inputs,
            outputs=self.outputs + other.outputs,
        )

    def __rshift__(self, other: "Graph") -> "Graph":
        return Graph(
            nodes=self.nodes + other.nodes,
            connections=self.connections + other.connections + tuple(GraphConnection(o, i) for o in self.outputs for i in other.inputs),
            inputs=self.inputs,
            outputs=other.outputs,
        )

    def add(self, nodes: tuple[GraphNode, ...] = (), connections: tuple[GraphConnection, ...] = (), inputs: tuple[GraphNodeId, ...] = (), outputs: tuple[GraphNodeId, ...] = ()) -> "Graph":
        return (self | Graph(nodes, connections, inputs, outputs))

    def is_valid(self):
        """
        Checks that all connections, inputs, and outputs only refer existing nodes.
        """
        node_id_set = set()
        node_stack = list(self.nodes)
        while len(node_stack):
            node = node_stack.pop()
            if node.id in node_id_set:
                continue

            if node.subgraph is not None:
                node_stack += node.subgraph.nodes

            node_id_set.add(node.id)

        references = set()
        references |= {c.from_node for c in self.connections}
        references |= {c.to_node for c in self.connections}
        references |= set(self.inputs)
        references |= set(self.outputs)
        return references <= node_id_set
