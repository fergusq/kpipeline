from kpipeline.pipeline import FilterPipe


def test_filter_pipe():
    predicate = lambda d, m: d > 15
    filter_pipe = FilterPipe(predicate)
    
    assert filter_pipe.apply([10, 20, 30], {}) == [20, 30]
