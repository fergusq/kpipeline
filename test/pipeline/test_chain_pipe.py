from test.pipeline.common import MultiplyPipe, SimplePipe


def test_chain_pipe():
    p1 = SimplePipe()
    p2 = MultiplyPipe()
    
    # Using chain method
    chain1 = p1.chain(p2)
    assert chain1.apply(10, {"offset": 5}) == 30 # (10+5)*2
    
    # Using | operator
    chain2 = p1 | p2
    assert chain2.apply(10, {"offset": 5}) == 30
    
    assert chain2.batch_apply([10, 20], {"offset": 5}) == [30, 50]
