from kpipeline.pipeline import BranchPipe
from test.pipeline.common import MultiplyPipe, SimplePipe


def test_branch_pipe():
    condition = lambda d, m: d > 15
    p_then = MultiplyPipe()
    p_else = SimplePipe()
    branch = BranchPipe(condition, p_then, p_else)
    
    assert branch.apply(20, {"offset": 5}) == 40 # 20 > 15 -> 20*2
    assert branch.apply(10, {"offset": 5}) == 15 # 10 <= 15 -> 10+5
    assert branch.batch_apply([20, 10], {"offset": 5}) == [40, 15]
