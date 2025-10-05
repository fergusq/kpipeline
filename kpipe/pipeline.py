from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Iterable, NamedTuple, Optional

from .graph import Graph, GraphNode, GraphConnection


class BasePipe[Input, Output, Metadata](ABC):
    """
    Abstract base class of all synchronous and asynchronous pipes.
    """

    @abstractmethod
    def apply(self, input: Input, metadata: Metadata) -> Output:
        """
        Apply the pipe to the given input data and return the result.
        """
        ...

    def chain[OtherOutput](self, other: "Pipe[Output, OtherOutput, Metadata]") -> "ChainPipe[Input, Output, OtherOutput, Metadata]":
        """
        Apply the second pipe to the result of the first pipe. Same as ChainPipe(self, other).
        """
        return ChainPipe(self, other)

    def to_node(self) -> GraphNode:
        return GraphNode(id=str(id(self)), type=type(self), title="Pipe")

    def to_graph(self) -> Graph:
        """
        Return the pipeline as a graph.
        """
        node = self.to_node()
        return Graph(nodes=(node,), connections=(), inputs=(node.id,), outputs=(node.id,))


class Pipe[Input, Output, Metadata](BasePipe[Input, Output, Metadata]):
    """
    Abstract base class of all synchronous pipes.
    """

    def __or__[OtherOutput](self, other: "Pipe[Output, OtherOutput, Metadata]") -> "ChainPipe[Input, Output, OtherOutput, Metadata]":
        """
        Alias for Pipe.chain.
        """
        return self.chain(other)


@dataclass(frozen=True)
class ChainPipe[Input, Middle, Output, Metadata](Pipe[Input, Output, Metadata]):
    """
    Connects to pipes so that the second pipe is applied to the result of the first pipe.
    """
    pipe1: BasePipe[Input, Middle, Metadata]
    pipe2: BasePipe[Middle, Output, Metadata]

    def apply(self, input: Input, metadata: Metadata) -> Output:
        return self.pipe2.apply(self.pipe1.apply(input, metadata), metadata)

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        return graph1 >> graph2


@dataclass(frozen=True)
class BranchPipe[Input, Output, Metadata](Pipe[Input, Output, Metadata]):
    """
    Selects one of two pipes to apply based on a condition.
    """
    condition: Callable[[Input, Metadata], bool]
    then_pipe: Pipe[Input, Output, Metadata]
    else_pipe: Pipe[Input, Output, Metadata]
    description: str = ""

    def apply(self, input: Input, metadata: Metadata) -> Output:
        if self.condition(input, metadata):
            return self.then_pipe.apply(input, metadata)

        else:
            return self.else_pipe.apply(input, metadata)

    def to_graph(self) -> Graph:
        condition_node = self.to_node()._replace(title=self.description or "Condition", shape="condition")
        then_graph = self.then_pipe.to_graph()
        else_graph = self.else_pipe.to_graph()
        return (
            (then_graph | else_graph)
            .add(nodes=(condition_node,))
            .add(connections=tuple(GraphConnection(condition_node.id, i, "yes") for i in then_graph.inputs))
            .add(connections=tuple(GraphConnection(condition_node.id, i, "no") for i in else_graph.inputs))
            ._replace(
                inputs=(condition_node.id,),
                outputs=then_graph.outputs + else_graph.outputs,
            )
        )


@dataclass(frozen=True)
class ConditionalPipe[InputOutput, Metadata](Pipe[InputOutput, InputOutput, Metadata]):
    """
    Applies the pipe only if the condition is met. Similar to BranchPipe except that the else branch is an identity function.
    Can only be used when the subpipe returns the same type that gets in.
    """
    condition: Callable[[InputOutput, Metadata], bool]
    subpipe: Pipe[InputOutput, InputOutput, Metadata]
    description: str = ""

    def apply(self, input: InputOutput, metadata: Metadata) -> InputOutput:
        if self.condition(input, metadata):
            return self.subpipe.apply(input, metadata)

        else:
            return input

    def to_graph(self) -> Graph:
        subpipe_graph = self.subpipe.to_graph()
        condition_node = self.to_node()._replace(title=self.description or "Condition", subgraph=subpipe_graph)
        return Graph(
            nodes=(condition_node,),
            connections=(),
            inputs=subpipe_graph.inputs,
            outputs=subpipe_graph.outputs,
        )


@dataclass(frozen=True)
class SelectPipe[Input, Output, Metadata, Key](Pipe[Input, Output, Metadata]):
    """
    Selects one of multiple pipes based on a selector function. BranchPipe generalized to more than two branches.
    """
    key: Callable[[Input, Metadata], Key]
    subpipes: Mapping[Key, Pipe[Input, Output, Metadata]]
    otherwise_pipe: Pipe[Input, Output, Metadata]
    description: str = ""

    def apply(self, input: Input, metadata: Metadata) -> Output:
        key = self.key(input, metadata)
        if key in self.subpipes:
            return self.subpipes[key].apply(input, metadata)

        else:
            return self.otherwise_pipe.apply(input, metadata)

    def to_graph(self) -> Graph:
        condition_node = self.to_node()._replace(title=self.description or "Condition", shape="condition")
        otherwise_graph = self.otherwise_pipe.to_graph()
        graph = otherwise_graph.add(
            nodes=(condition_node,),
            connections=tuple(GraphConnection(condition_node.id, i, "otherwise") for i in otherwise_graph.inputs),
        )
        for key, subpipe in self.subpipes.items():
            subpipe_graph = subpipe.to_graph()
            graph |= subpipe_graph
            graph = graph.add(connections=tuple(GraphConnection(condition_node.id, i, str(key)) for i in subpipe_graph.inputs))

        return graph._replace(inputs=(condition_node.id,))


@dataclass(frozen=True)
class ParallelPipe[Input, Output, CombinedOutput, Metadata](Pipe[Input, CombinedOutput, Metadata]):
    """
    Executes multiple pipes on the same input and combines the outputs.
    The synchronous implementation does not actually execute the pipes in parallel.
    """
    subpipes: Sequence[Pipe[Input, Output, Metadata]]
    combine: Callable[[Iterable[Output], Metadata], CombinedOutput]
    description: str = ""

    def apply(self, input: Input, metadata: Metadata) -> CombinedOutput:
        results = []
        for subpipe in self.subpipes:
            results.append(subpipe.apply(input, metadata))

        return self.combine(results, metadata)

    def to_graph(self) -> Graph:
        combine_node = self.to_node()._replace(title=self.description or "Combine results", shape="combine")
        graph = Graph(nodes=(combine_node,))
        for subpipe in self.subpipes:
            subpipe_graph = subpipe.to_graph()
            graph |= subpipe_graph
            graph = graph.add(connections=tuple(GraphConnection(o, combine_node.id) for o in subpipe_graph.outputs))

        return graph._replace(outputs=(combine_node.id,))


@dataclass(frozen=True)
class MetadataWrapperPipe[Input, Output, OuterMetadata, InnerMetadata](Pipe[Input, Output, OuterMetadata]):
    """
    Executes a pipe with changed metadata.
    """
    to_inner: Callable[[OuterMetadata], InnerMetadata]
    subpipe: Pipe[Input, Output, InnerMetadata]
    description: str = ""

    def apply(self, input: Input, metadata: OuterMetadata) -> Output:
        return self.subpipe.apply(input, self.to_inner(metadata))

    def to_graph(self) -> Graph:
        subpipe_graph = self.subpipe.to_graph()
        wrapper_node = self.to_node()._replace(title=self.description or "Wrap metadata", subgraph=subpipe_graph)
        return Graph(
            nodes=(wrapper_node,),
            connections=(),
            inputs=subpipe_graph.inputs,
            outputs=subpipe_graph.outputs,
        )

