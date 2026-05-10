import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
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


async def _batch_apply[I, O, M](p: SyncOrAsyncPipe[I, O, M], d: Sequence[I], m: M) -> Sequence[O]:
    if isinstance(p, AsyncPipe):
        return await p.async_batch_apply(d, m)
    
    else:
        return p.batch_apply(d, m)


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
        middle_batch = await _batch_apply(self.pipe1, data, metadata)
        return await _batch_apply(self.pipe2, middle_batch, metadata)

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
        output1, output2 = await asyncio.gather(
            _await_or_return(self.pipe1.apply(data, metadata)),
            _await_or_return(self.pipe2.apply(data, metadata)),
        )
        return self.merge(output1, output2)

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        batch1, batch2 = await asyncio.gather(
            _batch_apply(self.pipe1, data, metadata),
            _batch_apply(self.pipe2, data, metadata),
        )

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
        output1, output2, output3 = await asyncio.gather(
            _await_or_return(self.pipe1.apply(data, metadata)),
            _await_or_return(self.pipe2.apply(data, metadata)),
            _await_or_return(self.pipe3.apply(data, metadata)),
        )
        return await _await_or_return(self.merge(output1, output2, output3))

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        batch1, batch2, batch3 = await asyncio.gather(
            _batch_apply(self.pipe1, data, metadata),
            _batch_apply(self.pipe2, data, metadata),
            _batch_apply(self.pipe3, data, metadata),
        )

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

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        conds = await asyncio.gather(*[_call_or_apply(self.condition, d, metadata) for d in data])
        then_batch_future = _batch_apply(self.then_pipe, [d for c, d in zip(conds, data) if c], metadata)
        else_batch_future = _batch_apply(self.else_pipe, [d for c, d in zip(conds, data) if not c], metadata)
        then_batch, else_batch = await asyncio.gather(then_batch_future, else_batch_future)

        ans: list[Output] = []
        t = 0
        e = 0
        for cond in conds:
            if cond:
                ans.append(then_batch[t])
                t += 1
            
            else:
                ans.append(else_batch[e])
                e += 1

        return ans

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

    async def async_batch_apply(self, data: Sequence[InputOutput], metadata: Metadata) -> Sequence[InputOutput]:
        conds = await asyncio.gather(*[_call_or_apply(self.condition, d, metadata) for d in data])
        then_batch = await _batch_apply(self.subpipe, [d for c, d in zip(conds, data) if c], metadata)

        ans: list[InputOutput] = []
        t = 0
        for i, cond in enumerate(conds):
            if cond:
                ans.append(then_batch[t])
                t += 1
            
            else:
                ans.append(data[i])

        return ans

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

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        keys = await asyncio.gather(*[_call_or_apply(self.key, d, metadata) for d in data])
        unique_keys = list(set(keys))
        batches: dict[Key, list[tuple[int, Input]]] = {key: [] for key in unique_keys}
        for i, (key, item) in enumerate(zip(keys, data)):
            batches[key].append((i, item))
        
        batch_futures: list[Awaitable[Sequence[Output]]] = []
        for key in unique_keys:
            pipe = self.subpipes.get(key, self.otherwise_pipe)
            batch = [item for _, item in batches[key]]
            batch_futures.append(_batch_apply(pipe, batch, metadata))

        batch_results = await asyncio.gather(*batch_futures)
        batch_results_dict = {key: batch_results[i] for i, key in enumerate(unique_keys)}
        ans_dict = {i: result for key, batch in batches.items() for (i, _), result in zip(batch, batch_results_dict[key])}
        return [ans_dict[i] for i in range(len(data))]

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
        results: list[Awaitable[Output]] = []
        for subpipe in self.subpipes:
            results.append(_await_or_return(subpipe.apply(data, metadata)))

        gathered: Sequence[Output] = await asyncio.gather(*results)
        return await _call_or_apply(self.combine, gathered, metadata)

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[CombinedOutput]:
        results: list[Awaitable[Sequence[Output]]] = []
        for subpipe in self.subpipes:
            results.append(_batch_apply(subpipe, data, metadata))

        gathered = await asyncio.gather(*results)
        zipped_results: zip[Sequence[Output]] = zip(*gathered)
        return await asyncio.gather(*[_call_or_apply(self.combine, seq, metadata) for seq in zipped_results])

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

    async def async_batch_apply(self, data: Sequence[Sequence[Input]], metadata: Metadata) -> Sequence[Sequence[Output]]:
        combined_batch: list[Input] = []
        for batch in data:
            combined_batch += batch

        combined_results = await _batch_apply(self.subpipe, combined_batch, metadata)
        separated_results: list[Sequence[Output]] = []
        i = 0
        for batch in data:
            separated_results.append(combined_results[i:i+len(batch)])
            i += len(batch)

        return separated_results

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


@dataclass(frozen=True)
class AsyncFallbackPipe[Input, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Tries to execute the subpipe and on failure executes a fallback pipe instead.

    The batch_apply method of this class is configurable.
    If batching is enabled (enable_batching is True), the subpipe's async_batch_apply is called.
    In this case, the failure of a single item means the failure of the whole batch.
    If batching is not enabled, each item in the batch is executed separately and
    the failure of a single one does not mean the failure of the entire batch.
    """
    subpipe: SyncOrAsyncPipe[Input, Output, Metadata]
    fallback: SyncOrAsyncPipe[Input, Output, Metadata]
    exceptions: type | tuple[type, ...] = Exception
    enable_batching: bool = False
    description: str = "AsyncFallbackPipe"

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        try:
            return await _await_or_return(self.subpipe.apply(data, metadata))
        except Exception as e:
            if isinstance(e, self.exceptions):
                return await _await_or_return(self.fallback.apply(data, metadata))

            else:
                raise e

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        if self.enable_batching:
            try:
                return await _batch_apply(self.subpipe, data, metadata)
            except Exception as e:
                if isinstance(e, self.exceptions):
                    return await _batch_apply(self.fallback, data, metadata)

                else:
                    raise e

        else:
            # AsyncPipe.batch_apply() implementation just calls apply multiple times in sequence
            return await super().async_batch_apply(data, metadata)

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=f"{self.description} (enable_batching={self.enable_batching})")

    def is_wrapper(self) -> bool:
        return True

    def to_graph(self) -> Graph:
        graph = super().to_graph()
        graph_node = graph.nodes[0]  # Pipe.to_graph() only adds one node (with a subgraph)
        fallback_graph = self.fallback.to_graph()
        return (
            # This joins the graphs so that the inputs and outputs of both are
            # the inputs and outputs of the new graph
            (graph | fallback_graph)

            # Add a connection between the node containing the subpipe as a subgraph and the fallback graph
            .add(connections=tuple(
                GraphConnection(graph_node.id, fallback_input, label="On failure")
                for fallback_input in fallback_graph.inputs
            ))

            # Set the inputs to be the inputs of graph, so that fallback's inputs are
            # not inputs of the resulting graph
            ._replace(
                inputs=graph.inputs,
            )
        )


@dataclass
class AsyncBatchCollectorPipe[Input, Output, Metadata: Hashable](AsyncPipe[Input, Output, Metadata]):
    """
    Forms a batch from multiple inputs it receives within a time window and
    processes them together using the batch_apply of its subpipe.

    This is a **stateful** pipe unlike most other pipes.

    Usage:
        pipe = AsyncBatchCollectorPipe(subpipe=my_llm_pipe, time=0.05)
        # Multiple concurrent calls to pipe.apply() will be batched together
        # if they arrive within the time window.

    Lifecycle:
        Call `close()` when done to cancel the internal timer and process
        any remaining items. Can also be used as an async context manager.
    """

    subpipe: SyncOrAsyncPipe[Input, Output, Metadata]
    time: float
    max_batch_size: int | None = None
    description: str= "Accumulate batches of inputs"

    _accumulated_batch: list[tuple[Input, Metadata, asyncio.Future[Output]]] = field(init=False, default_factory=list)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _timer_task: asyncio.Task | None = field(init=False, default=None)
    _closed: bool = field(init=False, default=False)

    async def __aenter__(self) -> "AsyncBatchCollectorPipe[Input, Output, Metadata]":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Cancel the timer and flush any remaining accumulated items."""
        self._closed = True
        if self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

        await self._flush()

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        if self._closed:
            raise RuntimeError("AsyncBatchCollectorPipe is closed")

        future: asyncio.Future[Output] = asyncio.get_running_loop().create_future()

        async with self._lock:
            self._accumulated_batch.append((data, metadata, future))
            batch_len = len(self._accumulated_batch)

            # Start the timer on the first item in a new batch
            if self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(self._timer())

        # If we've hit the max batch size, flush immediately
        if self.max_batch_size is not None and batch_len >= self.max_batch_size:
            await self._flush()

        # Wait for the result
        return await future

    async def _timer(self) -> None:
        """Wait for the configured time window, then flush the batch."""
        try:
            await asyncio.sleep(self.time)
        except asyncio.CancelledError:
            return

        await self._flush()

    async def _flush(self) -> None:
        """Process all accumulated items."""
        async with self._lock:
            if not self._accumulated_batch:
                return

            # Grab the current batch and reset
            batch = self._accumulated_batch
            self._accumulated_batch = []

        # Group by metadata since batch_apply requires uniform metadata
        grouped_by_metadata: defaultdict[
            Metadata, list[tuple[Input, asyncio.Future[Output]]]
        ] = defaultdict(list)

        for data, metadata, future in batch:
            grouped_by_metadata[metadata].append((data, future))

        # Process each metadata group
        for metadata, items in grouped_by_metadata.items():
            inputs = [data for data, _ in items]
            futures = [future for _, future in items]

            try:
                results = await _batch_apply(self.subpipe, inputs, metadata)

                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)

            except Exception as e:
                # If batch processing fails, propagate the error to all waiters
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=f"{self.description} (time={self.time})")

    def is_wrapper(self) -> bool:
        return True


@dataclass
class AsyncSemaphorePipe[Input, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Limits the number of concurrent requests using a semaphore.
    Does not allow synchronous subpipes.

    The async_batch_apply of this class will split the incoming batch so
    that it fits to the limit if necessary.

    This is a **stateful** pipe unlike most other pipes.
    """
    subpipe: AsyncPipe[Input, Output, Metadata]
    max_concurrent: int
    description: str = "Limit number of concurrent inputs"

    _semaphore: asyncio.Semaphore = field(init=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        async with self._semaphore:
            return await self.subpipe.apply(data, metadata)

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        # If the data fits, process it normally
        if len(data) <= self.max_concurrent:
            async with self._lock:
                for _ in range(len(data)):
                    await self._semaphore.acquire()

            try:
                results = await self.subpipe.async_batch_apply(data, metadata)
                error = None
            except Exception as e:
                # We must release the lock even in case of failure, so error is not raised here
                error = e
                results = None

            # Lock is not required for releasing the semaphore
            for _ in range(len(data)):
                self._semaphore.release()

            if error is not None:
                raise error
            else:
                assert results is not None
                return results

        else:
            # If the data does not fit, split it to chunks of size max_concurrent
            chunk_results: list[Sequence[Output]] = []
            for i in range(0, len(data), self.max_concurrent):
                chunk = data[i:i+self.max_concurrent]
                chunk_results.append(await self.async_batch_apply(chunk, metadata))
                # There is no asyncio.gather here since the chunks wouldn't fit simultaneously anyway

            combined_results: list[Output] = []
            for chunk_result in chunk_results:
                combined_results += chunk_result

            return combined_results

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=f"{self.description} (max_concurrent={self.max_concurrent})")

    def is_wrapper(self) -> bool:
        return True


@dataclass(frozen=True)
class AsyncTimeoutPipe[Input, Output, Metadata](AsyncPipe[Input, Output, Metadata]):
    """
    Cancels the subpipe after a given amount of time (in seconds).
    Does not allow synchronous subpipes.

    The batch_apply method of this class is configurable.
    If batching is enabled (enable_batching is True), the subpipe's async_batch_apply is called.
    In this case, the timeout will cancel the whole batch.
    If batching is not enabled, each item in the batch is executed separately and
    have its own timeout.
    """
    subpipe: AsyncPipe[Input, Output, Metadata]
    timeout: float
    enable_batching: bool = False
    description: str = "Cancel after timeout"

    async def apply(self, data: Input, metadata: Metadata) -> Output:
        async with asyncio.timeout(self.timeout):
            return await self.subpipe.apply(data, metadata)

    async def async_batch_apply(self, data: Sequence[Input], metadata: Metadata) -> Sequence[Output]:
        if self.enable_batching:
            async with asyncio.timeout(self.timeout):
                return await self.subpipe.async_batch_apply(data, metadata)

        else:
            # AsyncPipe.batch_apply() implementation just calls apply multiple times in sequence
            return await super().async_batch_apply(data, metadata)

    def get_subgraph(self) -> Optional[Graph]:
        return self.subpipe.to_graph()

    def to_node(self) -> GraphNode:
        return super().to_node()._replace(title=f"{self.description} (timeout={self.timeout}, enable_batching={self.enable_batching})")

    def is_wrapper(self) -> bool:
        return True
