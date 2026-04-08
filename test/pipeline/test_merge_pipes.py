from kpipeline.pipeline import IdentityPipe, Merge2Pipe, Merge3Pipe
from test.pipeline.common import MultiplyPipe, SimplePipe


def test_merge2_pipe():
    p1 = SimplePipe()
    p2 = MultiplyPipe()
    merge = Merge2Pipe(p1, p2, lambda a, b: a + b)
    
    # (10+5) + (10*2) = 15 + 20 = 35
    assert merge.apply(10, {"offset": 5}) == 35
    assert merge.batch_apply([10, 20], {"offset": 5}) == [35, 65] # (15+20), (25+40)

def test_merge3_pipe():
    p1 = SimplePipe()
    p2 = MultiplyPipe()
    p3 = IdentityPipe[int, dict]()
    merge = Merge3Pipe(p1, p2, p3, lambda a, b, c: a + b + c)
    
    # (10+5) + (10*2) + 10 = 15 + 20 + 10 = 45
    assert merge.apply(10, {"offset": 5}) == 45
