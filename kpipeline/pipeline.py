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
    def apply(self, data: Input, metadata: Metadata) -> Output:
        """
        Apply the pipe to the given input data and return the result.
        """
        ...

    def batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        """
        Applies the pipe to multiple data points. By default merely calls apply in sequence.
        """
        return [self.apply(d, metadata) for d in data]

    def chain[OtherOutput](self, other: "Pipe[Output, OtherOutput, Metadata]") -> "ChainPipe[Input, Output, OtherOutput, Metadata]":
        """
        Apply the second pipe to the result of the first pipe. Same as ChainPipe(self, other).
        """
        return ChainPipe(self, other)

    def to_node(self) -> GraphNode:
        return GraphNode(id=str(id(self)), type=type(self), title="Pipe")

    def get_subgraph(self) -> Optional[Graph]:
        return None

    def is_wrapper(self) -> bool:
        return False

    def to_graph(self) -> Graph:
        """
        Return the pipeline as a graph.
        """
        if subgraph := self.get_subgraph():
            node = self.to_node()._replace(subgraph=subgraph)
            return Graph(
                nodes=(node,),
                connections=(),
                inputs=subgraph.inputs if self.is_wrapper() else (node.id,),
                outputs=subgraph.outputs if self.is_wrapper() else (node.id,),
            )

        else:
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


type PipeOrCallable[I, O, M] = Callable[[I, M], O] | BasePipe[I, O, M]


def _call_or_apply[I, O, M](f: PipeOrCallable[I, O, M], d: I, m: M) -> O:
    if isinstance(f, BasePipe):
        return f.apply(d, m)

    else:
        return f(d, m)


@dataclass(frozen=True)
class ChainPipe[Input, Middle, Output, Metadata](Pipe[Input, Output, Metadata]):
    """
    Connects to pipes so that the second pipe is applied to the result of the first pipe.
    """
    pipe1: BasePipe[Input, Middle, Metadata]
    pipe2: BasePipe[Middle, Output, Metadata]

    def apply(self, data: Input, metadata: Metadata) -> Output:
        return self.pipe2.apply(self.pipe1.apply(data, metadata), metadata)

    def batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        middle_batch = self.pipe1.batch_apply(data, metadata)
        return self.pipe2.batch_apply(middle_batch, metadata)

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        return graph1 >> graph2


@dataclass(frozen=True)
class IdentityPipe[InputOutput, Metadata](Pipe[InputOutput, InputOutput, Metadata]):
    """
    A pipe that returns its input.
    """
    def apply(self, data: InputOutput, metadata: Metadata) -> InputOutput:
        return data


@dataclass(frozen=True)
class Merge2Pipe[Input, Output1, Output2, Output, Metadata](Pipe[Input, Output, Metadata]):
    """
    Runs two pipes and merges their results.
    """
    pipe1: BasePipe[Input, Output1, Metadata]
    pipe2: BasePipe[Input, Output2, Metadata]
    merge: Callable[[Output1, Output2], Output]
    description: str = "Combine results"

    def apply(self, data: Input, metadata: Metadata) -> Output:
        output1 = self.pipe1.apply(data, metadata)
        output2 = self.pipe2.apply(data, metadata)
        return self.merge(output1, output2)

    def batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        batch1 = self.pipe1.batch_apply(data, metadata)
        batch2 = self.pipe2.batch_apply(data, metadata)
        assert len(data) == len(batch1) == len(batch2), "lengths do not match"
        return [self.merge(output1, output2) for output1, output2 in zip(batch1, batch2)]

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        combine_node = self.to_node()._replace(title=self.description, shape="combine")
        graph = Graph(nodes=(combine_node,), inputs=(combine_node.id,), outputs=(combine_node.id,))
        return (graph1 | graph2) >> graph


@dataclass(frozen=True)
class Merge3Pipe[Input, Output1, Output2, Output3, Output, Metadata](Pipe[Input, Output, Metadata]):
    """
    Runs three pipes and merges their results.
    """
    pipe1: BasePipe[Input, Output1, Metadata]
    pipe2: BasePipe[Input, Output2, Metadata]
    pipe3: BasePipe[Input, Output3, Metadata]
    merge: Callable[[Output1, Output2, Output3], Output]
    description: str = "Combine results"

    def apply(self, data: Input, metadata: Metadata) -> Output:
        output1 = self.pipe1.apply(data, metadata)
        output2 = self.pipe2.apply(data, metadata)
        output3 = self.pipe3.apply(data, metadata)
        return self.merge(output1, output2, output3)

    def batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        batch1 = self.pipe1.batch_apply(data, metadata)
        batch2 = self.pipe2.batch_apply(data, metadata)
        batch3 = self.pipe3.batch_apply(data, metadata)
        assert len(data) == len(batch1) == len(batch2) == len(batch3), "lengths do not match"
        return [self.merge(output1, output2, output3) for output1, output2, output3 in zip(batch1, batch2, batch3)]

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        graph3 = self.pipe3.to_graph()
        combine_node = self.to_node()._replace(title=self.description, shape="combine")
        graph = Graph(nodes=(combine_node,), inputs=(combine_node.id,), outputs=(combine_node.id,))
        return (graph1 | graph2 | graph3) >> graph


@dataclass(frozen=True)
class BranchPipe[Input, Output, Metadata](Pipe[Input, Output, Metadata]):
    """
    Selects one of two pipes to apply based on a condition.
    """
    condition: PipeOrCallable[Input, bool, Metadata]
    then_pipe: BasePipe[Input, Output, Metadata]
    else_pipe: BasePipe[Input, Output, Metadata]
    description: str = "Condition"

    def apply(self, data: Input, metadata: Metadata) -> Output:
        if _call_or_apply(self.condition, data, metadata):
            return self.then_pipe.apply(data, metadata)

        else:
            return self.else_pipe.apply(data, metadata)

    def to_graph(self) -> Graph:
        condition_node = self.to_node()._replace(title=self.description, shape="condition", subgraph=self.condition.to_graph() if isinstance(self.condition, Pipe) else None)
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
    subpipe: BasePipe[InputOutput, InputOutput, Metadata]
    description: str = "Condition"

    def apply(self, data: InputOutput, metadata: Metadata) -> InputOutput:
        if self.condition(data, metadata):
            return self.subpipe.apply(data, metadata)

        else:
            return data

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)

    def is_wrapper(self) -> bool:
        return True


@dataclass(frozen=True)
class SelectPipe[Input, Output, Metadata, Key](Pipe[Input, Output, Metadata]):
    """
    Selects one of multiple pipes based on a selector function. BranchPipe generalized to more than two branches.
    """
    key: PipeOrCallable[Input, Key, Metadata]
    subpipes: Mapping[Key, BasePipe[Input, Output, Metadata]]
    otherwise_pipe: BasePipe[Input, Output, Metadata]
    description: str = "Key"

    def apply(self, data: Input, metadata: Metadata) -> Output:
        key = _call_or_apply(self.key, data, metadata)
        if key in self.subpipes:
            return self.subpipes[key].apply(data, metadata)

        else:
            return self.otherwise_pipe.apply(data, metadata)

    def to_graph(self) -> Graph:
        condition_node = self.to_node()._replace(title=self.description, shape="condition", subgraph=self.key.to_graph() if isinstance(self.key, Pipe) else None)
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
    subpipes: Sequence[BasePipe[Input, Output, Metadata]]
    combine: PipeOrCallable[Sequence[Output], CombinedOutput, Metadata]
    description: str = "Combine results"

    def apply(self, data: Input, metadata: Metadata) -> CombinedOutput:
        results = []
        for subpipe in self.subpipes:
            results.append(subpipe.apply(data, metadata))

        seq: Sequence[Output] = results
        return _call_or_apply(self.combine, seq, metadata)

    def to_graph(self) -> Graph:
        combine_node = self.to_node()._replace(title=self.description, shape="combine", subgraph=self.combine.to_graph() if isinstance(self.combine, Pipe) else None)
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

    Does NOT allow to_inner to be a pipe: this is intentional, metadata is not transformed by pipes.
    """
    to_inner: Callable[[OuterMetadata], InnerMetadata]
    subpipe: BasePipe[Input, Output, InnerMetadata]
    description: str = "Wrap metadata"

    def apply(self, data: Input, metadata: OuterMetadata) -> Output:
        return self.subpipe.apply(data, self.to_inner(metadata))

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)

    def is_wrapper(self) -> bool:
        return True


@dataclass(frozen=True)
class MapPipe[Input, Output, Metadata](Pipe[Sequence[Input], Sequence[Output], Metadata]):
    """
    Maps a sequence of input objects into a sequence of output objects with a subpipe.

    Does NOT allow the subpipe to be a callable: this is intentional, functions that transforms Inputs to Outputs should be pipes.
    """
    subpipe: BasePipe[Input, Output, Metadata]
    description: str = "Map"

    def apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        return self.subpipe.batch_apply(data, metadata)

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)


@dataclass(frozen=True)
class FilterPipe[Input, Metadata](Pipe[Sequence[Input], Sequence[Input], Metadata]):
    """
    Filters a sequence of input object with a predicate.
    """
    predicate: PipeOrCallable[Input, bool, Metadata]
    description: str = "Filter"

    def apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Input]:
        return [i for i in data if _call_or_apply(self.predicate, i, metadata)]

    def get_subgraph(self) -> Optional[Graph]:
        return self.predicate.to_graph() if isinstance(self.predicate, Pipe) else None

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)


@dataclass(frozen=True)
class RetryPipe[Input, Output, Metadata](Pipe[Input, Output, Metadata]):
    """
    Tries to execute the subpipe several times in case it fails.
    """
    subpipe: BasePipe[Input, Output, Metadata]
    retries: int
    exceptions: type | tuple[type, ...]
    description: str = "Retry several times"

    def apply(self, data: Input, metadata: Metadata) -> Output:
        attempt = 0
        while True:
            try:
                return self.subpipe.apply(data, metadata)

            except Exception as e:
                if not isinstance(e, self.exceptions):
                    raise e

                attempt += 1
                if attempt > self.retries:
                    raise e

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)

    def is_wrapper(self) -> bool:
        return True

