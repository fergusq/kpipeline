from kpipeline.pipeline import IdentityPipe, SelectPipe
from test.pipeline.common import MultiplyPipe, SimplePipe


def test_select_pipe():
    def get_key(d, m): return "even" if d % 2 == 0 else "odd"
    
    p_even = MultiplyPipe()
    p_odd = SimplePipe()
    p_otherwise = IdentityPipe[int, dict]()
    
    select = SelectPipe(get_key, {"even": p_even, "odd": p_odd}, p_otherwise)
    
    assert select.apply(10, {"offset": 5}) == 20 # even -> 10*2
    assert select.apply(11, {"offset": 5}) == 16 # odd -> 11+5
    assert select.batch_apply([10, 11], {"offset": 5}) == [20, 16]
