import asyncio
import builtins
from typing import Iterable
import pytest

from kpipeline.graph import Graph, GraphNode, GraphConnection
from kpipeline.async_pipeline import (
    AsyncPipe,
    AsyncChainPipe,
    AsyncBranchPipe,
    AsyncConditionalPipe,
    AsyncSelectPipe,
    AsyncParallelPipe,
    AsyncMetadataWrapperPipe,
    AsyncMapPipe,
    AsyncFilterPipe,
    AsyncRetryPipe,
)

from .test_pipeline import MulPipe, FailingPipe


# ----------------------------------------------------------------------
# Helper concrete async Pipe implementations
# ----------------------------------------------------------------------


class AsyncAddPipe(AsyncPipe[int, int, None]):  # type: ignore[misc]
    """Async version of ``AddPipe`` – adds 1 to the incoming integer."""

    async def apply(self, input: int, metadata: None) -> int:  # noqa: D401
        return input + 1


class AsyncMulPipe(AsyncPipe[int, int, None]):  # type: ignore[misc]
    """Async version of ``MulPipe`` – multiplies the incoming integer by 2."""

    async def apply(self, input: int, metadata: None) -> int:  # noqa: D401
        return input * 2


# ----------------------------------------------------------------------
# Async chaining (`AsyncPipe.async_chain` / `AsyncPipe.__or__`)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_chain_operator_produces_asyncchainpipe_and_composes_results():
    add = AsyncAddPipe()
    mul = AsyncMulPipe()

    # Explicit method
    chain1 = add.async_chain(mul)
    assert isinstance(chain1, AsyncChainPipe)

    # ``|`` operator (syntactic sugar)
    chain2 = add | mul
    assert isinstance(chain2, AsyncChainPipe)

    for chain in (chain1, chain2):
        # (x + 1) * 2
        result = await chain.apply(3, None)
        assert result == (3 + 1) * 2

        # Graph representation must be a valid graph containing two nodes
        g = chain.to_graph()
        assert isinstance(g, Graph)
        assert len(g.nodes) == 2
        assert g.is_valid()


# ----------------------------------------------------------------------
# AsyncBranchPipe – conditional execution of one of two sub‑pipes
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_branch_pipe_applies_correct_branch():
    # condition is async (simulated with ``asyncio.sleep``)
    async def cond(x: int, _: None) -> bool:
        await asyncio.sleep(0)  # force a real await point
        return x > 0

    branch = AsyncBranchPipe(
        condition=cond,
        then_pipe=AsyncAddPipe(),
        else_pipe=AsyncMulPipe(),
        description="pos?",
    )

    # Positive numbers go through ``then_pipe`` (AsyncAddPipe)
    assert await branch.apply(5, None) == 6
    # Zero / negative numbers go through ``else_pipe`` (AsyncMulPipe)
    assert await branch.apply(-2, None) == -4
    assert await branch.apply(0, None) == 0

    g = branch.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()
    # Exactly one condition node should be present
    condition_nodes = [n for n in g.nodes if n.title == "pos?" or n.shape == "condition"]
    assert len(condition_nodes) == 1


# ----------------------------------------------------------------------
# AsyncConditionalPipe – apply a pipe only when a predicate holds, otherwise identity
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_conditional_pipe_behaviour():
    async def is_even(x: int, _: None) -> bool:
        return x % 2 == 0

    cond = AsyncConditionalPipe(
        condition=is_even,
        subpipe=AsyncMulPipe(),
        description="even?",
    )

    # Even numbers are multiplied
    assert await cond.apply(4, None) == 8
    # Odd numbers are returned unchanged
    assert await cond.apply(5, None) == 5

    g = cond.to_graph()
    assert isinstance(g, Graph)
    assert any(n.title == "even?" for n in g.nodes)
    assert g.is_valid()


# ----------------------------------------------------------------------
# AsyncSelectPipe – route to one of many sub‑pipes based on a selector key
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_select_pipe_routing_and_graph():
    subpipes = {
        0: AsyncAddPipe(),
        1: AsyncMulPipe(),
    }

    async def selector(x: int, _: None) -> int:
        # async selector just to prove awaiting works
        await asyncio.sleep(0)
        return x % 3

    select = AsyncSelectPipe(
        key=selector,
        subpipes=subpipes,
        otherwise_pipe=AsyncAddPipe(),
        description="mod‑3",
    )

    # 0 -> key 0 -> AsyncAddPipe
    assert await select.apply(6, None) == 7
    # 1 -> key 1 -> AsyncMulPipe
    assert await select.apply(4, None) == 8
    # 2 -> key 2 not in mapping -> otherwise (AsyncAddPipe)
    assert await select.apply(2, None) == 3

    g = select.to_graph()
    assert isinstance(g, Graph)
    assert any(n.title == "mod‑3" for n in g.nodes)
    assert g.is_valid()


# ----------------------------------------------------------------------
# AsyncParallelPipe – run several pipes in parallel and combine the results
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_parallel_pipe_combines_results():
    # combine function just sums the intermediate results (may be async)
    async def combiner(values: Iterable[int], _: None) -> int:
        return sum(values)

    parallel = AsyncParallelPipe(
        subpipes=[AsyncAddPipe(), AsyncMulPipe()],  # (x+1) and (x*2)
        combine=combiner,
        description="sum‑parallel",
    )

    # (3+1) + (3*2) = 4 + 6 = 10
    assert await parallel.apply(3, None) == 10

    g = parallel.to_graph()
    assert isinstance(g, Graph)
    combine_nodes = [n for n in g.nodes if n.title == "sum‑parallel" or n.shape == "combine"]
    assert len(combine_nodes) == 1
    assert g.is_valid()


# ----------------------------------------------------------------------
# AsyncMetadataWrapperPipe – transform outer metadata before delegating to a sub‑pipe
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_metadata_wrapper_pipe_transforms_metadata():
    inner = AsyncMulPipe()

    # outer metadata is a dict; we extract the value under key "factor"
    async def to_inner(outer: dict) -> None:
        # The inner ``MulPipe`` ignores its metadata, but we still await to prove the plumbing works
        await asyncio.sleep(0)
        return outer["factor"]

    wrapper = AsyncMetadataWrapperPipe(
        to_inner=to_inner,
        subpipe=inner,
        description="unwrap‑factor",
    )

    outer_meta = {"factor": None}
    # ``inner`` multiplies by 2, metadata is irrelevant
    assert await wrapper.apply(5, outer_meta) == 10

    g = wrapper.to_graph()
    assert isinstance(g, Graph)
    assert any(n.title == "unwrap‑factor" for n in g.nodes)
    assert g.is_valid()


# ----------------------------------------------------------------------
# Edge‑case: chaining more than two pipes works recursively (mix sync & async)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_async_chain_composition():
    # Mix a sync ``Pipe`` with async ones to ensure the generic union works
    chain = AsyncAddPipe().async_chain(MulPipe()).async_chain(AsyncAddPipe())  # ((x+1)*2)+1
    assert isinstance(chain, AsyncChainPipe)

    result = await chain.apply(2, None)
    assert result == ((2 + 1) * 2) + 1  # 7

    g = chain.to_graph()
    assert len(g.nodes) == 3
    assert g.is_valid()


# ----------------------------------------------------------------------
# Ensure that the abstract base class ``AsyncPipe`` cannot be instantiated directly
# ----------------------------------------------------------------------


def test_async_pipe_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AsyncPipe()  # type: ignore


# ----------------------------------------------------------------------
# AsyncMapPipe
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_map_pipe():
    pipe = AsyncMapPipe(MulPipe())
    result = await pipe.apply([1, 2], None)
    assert result == [2, 4]

    g = pipe.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()


# ----------------------------------------------------------------------
# AsyncFilterPipe
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_filter_pipe():
    async def async_pred(x, m):
        await asyncio.sleep(0)
        return x > 0

    pipe = AsyncFilterPipe(predicate=async_pred)
    result = await pipe.apply([-1, 0, 5, 2], None)
    assert result == [5, 2]

    g = pipe.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()


# ----------------------------------------------------------------------
# AsyncRetryPipe
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_async_retry_pipe_successful_retry():
    async_pipe = AsyncRetryPipe(subpipe=FailingPipe(), retries=2, exceptions=RuntimeError)
    # Same logic as the sync test – first attempt fails, second succeeds
    assert await async_pipe.apply(7, None) == 14

    g = async_pipe.to_graph()
    assert isinstance(g, Graph)
    assert g.is_valid()


@pytest.mark.asyncio
async def test_async_retry_pipe_exhausts():
    class AsyncAlwaysFail(AsyncPipe[int, int, None]):  # type: ignore[misc]
        async def apply(self, input: int, metadata: None) -> int:
            raise RuntimeError("always fail")

        def to_graph(self) -> Graph:  # pragma: no cover
            node = self.to_node()
            return Graph(nodes=(node,), connections=(), inputs=(node.id,), outputs=(node.id,))

    pipe = AsyncRetryPipe(subpipe=AsyncAlwaysFail(), retries=0, exceptions=RuntimeError)
    with pytest.raises(RuntimeError):
        await pipe.apply(42, None)

