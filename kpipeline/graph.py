from typing import Literal, NamedTuple, Optional, Type

type GraphNodeId = str
type GraphNodeShape = Literal["default", "combine", "condition"]
"""
The shape of the node.
`combine` is used for nodes that combine multiple inputs into one output.
`condition` is used for nodes that diverge path based on condition.
The actual shape depends on the rendering backend.
"""


class GraphNode(NamedTuple):
    id: GraphNodeId
    title: str
    type: Optional[Type] = None
    shape: GraphNodeShape = "default"
    subgraph: "Optional[Graph]" = None

    def to_dot(self, indent: int = 0) -> str:
        shape_map = {
            "default": "box",
            "combine": "ellipse",
            "condition": "diamond",
        }
        dot_shape = shape_map.get(self.shape, "box")
        node_def = f"""{"  "*indent}"{self.id}" [label="{self.title}", shape={dot_shape}];"""
        if self.subgraph:
            sub_dot = self.subgraph.to_dot(name=None, indent=indent+1)
            node_def += f"""{"  "*indent}subgraph "cluster_{self.id}" {{"""
            node_def += f"""{"  "*indent}  label="{self.title}";"""
            node_def += sub_dot
            node_def += "  "*indent + "}"
        return node_def


class GraphConnection(NamedTuple):
    from_node: GraphNodeId
    to_node: GraphNodeId
    label: str = ""

    def to_dot(self, indent: int = 0) -> str:
        label_part = f' [label="{self.label}"]' if self.label else ""
        return f"""{"  "*indent}"{self.from_node}" -> "{self.to_node}"{label_part};"""


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

    def to_dot(self, name: str | None = "graph", indent: int = 0) -> str:
        lines: list[str] = []
        if name is not None:
            lines.append(f"digraph {name} {{")

        for node in self.nodes:
            lines.append(node.to_dot(indent=indent+1))

        for conn in self.connections:
            lines.append(conn.to_dot(indent=indent+1))

        if name is not None:
            lines.append("}")

        return "\n".join(lines)
