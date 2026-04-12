from kpipeline.graph import Graph, GraphNode, GraphConnection


def test_simple_graph_dot():
    node1 = GraphNode(id="n1", title="Node 1")
    node2 = GraphNode(id="n2", title="Node 2")
    conn = GraphConnection(from_node="n1", to_node="n2", label="connect")
    graph = Graph(nodes=(node1, node2), connections=(conn,))
    
    dot = graph.to_dot()
    assert 'digraph g {' in dot
    assert '"n1" [label="Node 1", shape=box];' in dot
    assert '"n2" [label="Node 2", shape=box];' in dot
    assert '"n1" -> "n2" [label="connect"];' in dot

def test_shapes_dot():
    node_combine = GraphNode(id="nc", title="Combine", shape="combine")
    node_cond = GraphNode(id="nd", title="Condition", shape="condition")
    graph = Graph(nodes=(node_combine, node_cond))
    
    dot = graph.to_dot()
    assert '"nc" [label="Combine", shape=ellipse];' in dot
    assert '"nd" [label="Condition", shape=diamond];' in dot

def test_subgraph_dot():
    inner_node = GraphNode(id="in", title="Inner")
    inner_graph = Graph(nodes=(inner_node,))
    
    outer_node = GraphNode(id="out", title="Outer", subgraph=inner_graph)
    graph = Graph(nodes=(outer_node,))
    
    dot = graph.to_dot()
    assert 'subgraph "cluster_out" {' in dot
    assert 'label="Outer";' in dot
    assert '"in" [label="Inner", shape=box];' in dot
