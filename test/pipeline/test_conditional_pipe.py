from kpipeline.pipeline import ConditionalPipe
from test.pipeline.common import MultiplyPipe


def test_conditional_pipe():
    condition = lambda d, m: d % 2 == 0
    subpipe = MultiplyPipe()
    cond_pipe = ConditionalPipe(condition, subpipe)
    
    assert cond_pipe.apply(10, {}) == 20 # even -> 10*2
    assert cond_pipe.apply(11, {}) == 11 # odd -> 11
    assert cond_pipe.batch_apply([10, 11], {}) == [20, 11]
