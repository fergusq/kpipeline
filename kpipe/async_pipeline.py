import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Iterable, NamedTuple, Optional

from .graph import Graph, GraphNode, GraphConnection
from .pipeline import BasePipe, Pipe


class AsyncPipe[Input, Output, Metadata](BasePipe[Input, Awaitable[Output], Metadata]):
    """
    Abstract base class of all asynchronous pipes.
    """

    def async_chain[OtherOutput](self, other: "Pipe[Output, OtherOutput, Metadata] | AsyncPipe[Output, OtherOutput, Metadata]") -> "AsyncChainPipe[Input, Output, OtherOutput, Metadata]":
        """
        Apply the second pipe to the result of the first pipe. Same as AsyncChainPipe(self, other).
        """
        return AsyncChainPipe(self, other)

    def __or__[OtherOutput](self, other: "Pipe[Output, OtherOutput, Metadata] | AsyncPipe[Output, OtherOutput, Metadata]") -> "AsyncChainPipe[Input, Output, OtherOutput, Metadata]":
        """
        Alias for AsyncPipe.async_chain.
        """
        return self.async_chain(other)


type SyncOrAsyncPipe[I, O, M] = Pipe[I, O, M] | AsyncPipe[I, O, M]


async def _await_or_return[T](data: T | Awaitable[T]) -> T:
    if isinstance(data, Awaitable):
        return await data

    else:
        return data


@dataclass(frozen=True)
class AsyncChainPipe[Input, Middle, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Connects to pipes so that the second pipe is applied to the result of the first pipe.
    """
    pipe1: SyncOrAsyncPipe[Input, Middle, Metadata]
    pipe2: SyncOrAsyncPipe[Middle, Output, Metadata]

    async def apply(self, input: Input, metadata: Metadata) -> Output:
        middle = await _await_or_return(self.pipe1.apply(input, metadata))
        return await _await_or_return(self.pipe2.apply(middle, metadata))

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        return graph1 >> graph2


@dataclass(frozen=True)
class AsyncBranchPipe[Input, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Selects one of two pipes to apply based on a condition.
    """
    condition: Callable[[Input, Metadata], bool | Awaitable[bool]]
    then_pipe: SyncOrAsyncPipe[Input, Output, Metadata]
    else_pipe: SyncOrAsyncPipe[Input, Output, Metadata]
    description: str = ""

    async def apply(self, input: Input, metadata: Metadata) -> Output:
        if await _await_or_return(self.condition(input, metadata)):
            return await _await_or_return(self.then_pipe.apply(input, metadata))

        else:
            return await _await_or_return(self.else_pipe.apply(input, metadata))

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
class AsyncConditionalPipe[InputOutput, Metadata](AsyncPipe[InputOutput, InputOutput, Metadata]):
    """
    Applies the pipe only if the condition is met. Similar to AsyncBranchPipe except that the else branch is an identity function.
    Can only be used when the subpipe returns the same type that gets in.
    """
    condition: Callable[[InputOutput, Metadata], bool | Awaitable[bool]]
    subpipe: SyncOrAsyncPipe[InputOutput, InputOutput, Metadata]
    description: str = ""

    async def apply(self, input: InputOutput, metadata: Metadata) -> InputOutput:
        if await _await_or_return(self.condition(input, metadata)):
            return await _await_or_return(self.subpipe.apply(input, metadata))

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
class AsyncSelectPipe[Input, Output, Metadata, Key](AsyncPipe[Input, Output, Metadata]):
    """
    Selects one of multiple pipes based on a selector function. AsyncBranchPipe generalized to more than two branches.
    """
    key: Callable[[Input, Metadata], Key | Awaitable[Key]]
    subpipes: Mapping[Key, SyncOrAsyncPipe[Input, Output, Metadata]]
    otherwise_pipe: SyncOrAsyncPipe[Input, Output, Metadata]
    description: str = ""

    async def apply(self, input: Input, metadata: Metadata) -> Output:
        key = await _await_or_return(self.key(input, metadata))
        if key in self.subpipes:
            return await _await_or_return(self.subpipes[key].apply(input, metadata))

        else:
            return await _await_or_return(self.otherwise_pipe.apply(input, metadata))

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
class AsyncParallelPipe[Input, Output, CombinedOutput, Metadata](AsyncPipe[Input, CombinedOutput, Metadata]):
    """
    Executes multiple pipes in parallel on the same input and combines the outputs.
    """
    subpipes: Sequence[SyncOrAsyncPipe[Input, Output, Metadata]]
    combine: Callable[[Iterable[Output], Metadata], CombinedOutput | Awaitable[CombinedOutput]]
    description: str = ""

    async def apply(self, input: Input, metadata: Metadata) -> CombinedOutput:
        results = []
        for subpipe in self.subpipes:
            results.append(_await_or_return(subpipe.apply(input, metadata)))

        return await _await_or_return(self.combine(await asyncio.gather(*results), metadata))

    def to_graph(self) -> Graph:
        combine_node = self.to_node()._replace(title=self.description or "Combine results", shape="combine")
        graph = Graph(nodes=(combine_node,))
        for subpipe in self.subpipes:
            subpipe_graph = subpipe.to_graph()
            graph |= subpipe_graph
            graph = graph.add(connections=tuple(GraphConnection(o, combine_node.id) for o in subpipe_graph.outputs))

        return graph._replace(outputs=(combine_node.id,))


@dataclass(frozen=True)
class AsyncMetadataWrapperPipe[Input, Output, OuterMetadata, InnerMetadata](AsyncPipe[Input, Output, OuterMetadata]):
    """
    Executes a pipe with changed metadata.
    """
    to_inner: Callable[[OuterMetadata], InnerMetadata | Awaitable[InnerMetadata]]
    subpipe: SyncOrAsyncPipe[Input, Output, InnerMetadata]
    description: str = ""

    async def apply(self, input: Input, metadata: OuterMetadata) -> Output:
        return await _await_or_return(self.subpipe.apply(input, await _await_or_return(self.to_inner(metadata))))

    def to_graph(self) -> Graph:
        subpipe_graph = self.subpipe.to_graph()
        wrapper_node = self.to_node()._replace(title=self.description or "Wrap metadata", subgraph=subpipe_graph)
        return Graph(
            nodes=(wrapper_node,),
            connections=(),
            inputs=subpipe_graph.inputs,
            outputs=subpipe_graph.outputs,
        )

