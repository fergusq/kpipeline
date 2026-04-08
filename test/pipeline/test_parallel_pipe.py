from kpipeline.pipeline import ParallelPipe
from test.pipeline.common import MultiplyPipe, SimplePipe


def test_parallel_pipe():
    p1 = SimplePipe()
    p2 = MultiplyPipe()
    combine = lambda results, m: sum(results)
    
    parallel = ParallelPipe([p1, p2], combine)
    
    # (10+5) + (10*2) = 15 + 20 = 35
    assert parallel.apply(10, {"offset": 5}) == 35
    assert parallel.batch_apply([10, 20], {"offset": 5}) == [35, 65]
