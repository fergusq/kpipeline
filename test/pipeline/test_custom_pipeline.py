from test.pipeline.common import SimplePipe


def test_simple_pipe():
    pipe = SimplePipe()
    assert pipe.apply(10, {"offset": 5}) == 15
    assert pipe.batch_apply([10, 20], {"offset": 5}) == [15, 25]
