import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Optional

from .graph import Graph, GraphNode, GraphConnection
from .pipeline import BasePipe, Pipe


class AsyncPipe[Input, Output, Metadata](BasePipe[Input, Awaitable[Output], Metadata]):
    """
    Abstract base class of all asynchronous pipes.
    """

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        """
        Like batch_apply, but returns Awaitable[Sequence[Output]] instead of Sequence[Awaitable[Output]].
        The default implementation uses asyncio.gather.
        """
        return await asyncio.gather(*self.batch_apply(data, metadata))

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


type SyncOrAsyncPipeOrCallable[I, O, M] = Callable[[I, M], O] | Callable[[I, M], Awaitable[O]] | SyncOrAsyncPipe[I, O, M]


async def _call_or_apply[I, O, M](f: SyncOrAsyncPipeOrCallable[I, O, M], d: I, m: M) -> O:
    if isinstance(f, BasePipe):
        return await _await_or_return(f.apply(d, m))

    else:
        return await _await_or_return(f(d, m))


@dataclass(frozen=True)
class AsyncChainPipe[Input, Middle, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Connects to pipes so that the second pipe is applied to the result of the first pipe.
    """
    pipe1: SyncOrAsyncPipe[Input, Middle, Metadata]
    pipe2: SyncOrAsyncPipe[Middle, Output, Metadata]

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        middle = await _await_or_return(self.pipe1.apply(data, metadata))
        return await _await_or_return(self.pipe2.apply(middle, metadata))

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        if isinstance(self.pipe1, AsyncPipe):
            middle_batch = await self.pipe1.async_batch_apply(data, metadata)

        else:
            middle_batch = self.pipe1.batch_apply(data, metadata)

        if isinstance(self.pipe2, AsyncPipe):
            return await self.pipe2.async_batch_apply(middle_batch, metadata)

        else:
            return self.pipe2.batch_apply(middle_batch, metadata)

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        return graph1 >> graph2


@dataclass(frozen=True)
class AsyncIdentityPipe[InputOutput, Metadata](AsyncPipe[InputOutput, InputOutput, Metadata]):
    """
    A pipe that returns its input.
    """
    async def apply(self, data: InputOutput, metadata: Metadata) -> InputOutput:
        return data


@dataclass(frozen=True)
class AsyncMerge2Pipe[Input, Output1, Output2, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Runs two pipes and merges their results.
    """
    pipe1: SyncOrAsyncPipe[Input, Output1, Metadata]
    pipe2: SyncOrAsyncPipe[Input, Output2, Metadata]
    merge: Callable[[Output1, Output2], Output]
    description: str = "Combine results"

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        output1 = await _await_or_return(self.pipe1.apply(data, metadata))
        output2 = await _await_or_return(self.pipe2.apply(data, metadata))
        return self.merge(output1, output2)

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        if isinstance(self.pipe1, AsyncPipe):
            batch1 = await self.pipe1.async_batch_apply(data, metadata)

        else:
            batch1 = self.pipe1.batch_apply(data, metadata)

        if isinstance(self.pipe2, AsyncPipe):
            batch2 = await self.pipe2.async_batch_apply(data, metadata)

        else:
            batch2 = self.pipe2.batch_apply(data, metadata)

        assert len(data) == len(batch1) == len(batch2), "lengths do not match"
        return await asyncio.gather(*[_await_or_return(self.merge(output1, output2)) for output1, output2 in zip(batch1, batch2)])

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        combine_node = self.to_node()._replace(title=self.description, shape="combine")
        graph = Graph(nodes=(combine_node,), inputs=(combine_node.id,), outputs=(combine_node.id,))
        return (graph1 | graph2) >> graph


@dataclass(frozen=True)
class AsyncMerge3Pipe[Input, Output1, Output2, Output3, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Runs three pipes and merges their results.
    """
    pipe1: SyncOrAsyncPipe[Input, Output1, Metadata]
    pipe2: SyncOrAsyncPipe[Input, Output2, Metadata]
    pipe3: SyncOrAsyncPipe[Input, Output3, Metadata]
    merge: Callable[[Output1, Output2, Output3], Output] | Callable[[Output1, Output2, Output3], Awaitable[Output]]
    description: str = "Combine results"

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        output1 = await _await_or_return(self.pipe1.apply(data, metadata))
        output2 = await _await_or_return(self.pipe2.apply(data, metadata))
        output3 = await _await_or_return(self.pipe3.apply(data, metadata))
        return await _await_or_return(self.merge(output1, output2, output3))

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        if isinstance(self.pipe1, AsyncPipe):
            batch1 = await self.pipe1.async_batch_apply(data, metadata)

        else:
            batch1 = self.pipe1.batch_apply(data, metadata)

        if isinstance(self.pipe2, AsyncPipe):
            batch2 = await self.pipe2.async_batch_apply(data, metadata)

        else:
            batch2 = self.pipe2.batch_apply(data, metadata)

        if isinstance(self.pipe3, AsyncPipe):
            batch3 = await self.pipe3.async_batch_apply(data, metadata)

        else:
            batch3 = self.pipe3.batch_apply(data, metadata)

        assert len(data) == len(batch1) == len(batch2) == len(batch3), "lengths do not match"
        return await asyncio.gather(*[_await_or_return(self.merge(output1, output2, output3)) for output1, output2, output3 in zip(batch1, batch2, batch3)])

    def to_graph(self) -> Graph:
        graph1 = self.pipe1.to_graph()
        graph2 = self.pipe2.to_graph()
        graph3 = self.pipe3.to_graph()
        combine_node = self.to_node()._replace(title=self.description, shape="combine")
        graph = Graph(nodes=(combine_node,), inputs=(combine_node.id,), outputs=(combine_node.id,))
        return (graph1 | graph2 | graph3) >> graph


@dataclass(frozen=True)
class AsyncBranchPipe[Input, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Selects one of two pipes to apply based on a condition.
    """
    condition: SyncOrAsyncPipeOrCallable[Input, bool, Metadata]
    then_pipe: SyncOrAsyncPipe[Input, Output, Metadata]
    else_pipe: SyncOrAsyncPipe[Input, Output, Metadata]
    description: str = ""

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        if await _call_or_apply(self.condition, data, metadata):
            return await _await_or_return(self.then_pipe.apply(data, metadata))

        else:
            return await _await_or_return(self.else_pipe.apply(data, metadata))

    def to_graph(self) -> Graph:
        condition_node = self.to_node()._replace(title=self.description or "Condition", shape="condition", subgraph=self.condition.to_graph() if isinstance(self.condition, Pipe) else None)
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

    async def apply(self, data: InputOutput, metadata: Metadata) -> InputOutput:
        if await _await_or_return(self.condition(data, metadata)):
            return await _await_or_return(self.subpipe.apply(data, metadata))

        else:
            return data

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)

    def is_wrapper(self) -> bool:
        return True


@dataclass(frozen=True)
class AsyncSelectPipe[Input, Output, Metadata, Key](AsyncPipe[Input, Output, Metadata]):
    """
    Selects one of multiple pipes based on a selector function. AsyncBranchPipe generalized to more than two branches.
    """
    key: SyncOrAsyncPipeOrCallable[Input, Key, Metadata]
    subpipes: Mapping[Key, SyncOrAsyncPipe[Input, Output, Metadata]]
    otherwise_pipe: SyncOrAsyncPipe[Input, Output, Metadata]
    description: str = ""

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        key = await _call_or_apply(self.key, data, metadata)
        if key in self.subpipes:
            return await _await_or_return(self.subpipes[key].apply(data, metadata))

        else:
            return await _await_or_return(self.otherwise_pipe.apply(data, metadata))

    def to_graph(self) -> Graph:
        condition_node = self.to_node()._replace(title=self.description or "Condition", shape="condition", subgraph=self.key.to_graph() if isinstance(self.key, Pipe) else None)
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
    combine: SyncOrAsyncPipeOrCallable[Sequence[Output], CombinedOutput, Metadata]
    description: str = ""

    async def apply(self, data: Input, metadata: Metadata) -> CombinedOutput:
        results = []
        for subpipe in self.subpipes:
            results.append(_await_or_return(subpipe.apply(data, metadata)))

        gathered: Sequence[Output] = await asyncio.gather(*results)
        return await _call_or_apply(self.combine, gathered, metadata)

    def to_graph(self) -> Graph:
        combine_node = self.to_node()._replace(title=self.description or "Combine results", shape="combine", subgraph=self.combine.to_graph() if isinstance(self.combine, Pipe) else None)
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

    Does NOT allow to_inner to be a pipe: this is intentional, metadata is not transformed by pipes.
    """
    to_inner: Callable[[OuterMetadata], InnerMetadata | Awaitable[InnerMetadata]]
    subpipe: SyncOrAsyncPipe[Input, Output, InnerMetadata]
    description: str = ""

    async def apply(self, data: Input, metadata: OuterMetadata) -> Output:
        return await _await_or_return(self.subpipe.apply(data, await _await_or_return(self.to_inner(metadata))))

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)

    def is_wrapper(self) -> bool:
        return True


@dataclass(frozen=True)
class AsyncMapPipe[Input, Output, Metadata](AsyncPipe[Sequence[Input], Sequence[Output], Metadata]):
    """
    Maps a sequence of input objects into a sequence of output objects with a subpipe.

    Does NOT allow the subpipe to be a callable: this is intentional, functions that transforms Inputs to Outputs should be pipes.
    """
    subpipe: SyncOrAsyncPipe[Input, Output, Metadata]
    description: str = "Map"

    async def apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        if isinstance(self.subpipe, AsyncPipe):
            return await self.subpipe.async_batch_apply(data, metadata)

        else:
            return self.subpipe.batch_apply(data, metadata)

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)


@dataclass(frozen=True)
class AsyncFilterPipe[Input, Metadata](AsyncPipe[Sequence[Input], Sequence[Input], Metadata]):
    """
    Filters a sequence of input object with a predicate.
    """
    predicate: SyncOrAsyncPipeOrCallable[Input, bool, Metadata]
    description: str = "Filter"

    async def apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Input]:
        return [i for i in data if await _call_or_apply(self.predicate, i, metadata)]

    def get_subgraph(self) -> Optional[Graph]:
        return self.predicate.to_graph() if isinstance(self.predicate, BasePipe) else None

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=self.description)


@dataclass(frozen=True)
class AsyncRetryPipe[Input, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Tries to execute the subpipe several times in case it fails.
    """
    subpipe: SyncOrAsyncPipe[Input, Output, Metadata]
    retries: int
    exceptions: type | tuple[type, ...]
    description: str = "Retry several times"

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        attempt = 0
        while True:
            try:
                return await _await_or_return(self.subpipe.apply(data, metadata))

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

