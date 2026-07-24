import pytest

from kpipeline.pipeline import Parallel2Pipe
from test.pipeline.common import SimplePipe, ToStringPipe


def test_async_parallel2_pipe():
    p1 = SimplePipe()
    p2 = ToStringPipe()
    combine = lambda results, m: f"{repr(results[0])} {repr(results[1])}"
    
    parallel = Parallel2Pipe(p1, p2, combine)
    
    assert parallel.apply(10, {"offset": 5}) == "15 '10'"
    assert parallel.batch_apply([10, 20], {"offset": 5}) == ["15 '10'", "25 '20'"]
