from kpipeline.pipeline import MapPipe
from test.pipeline.common import SimplePipe


def test_map_pipe():
    subpipe = SimplePipe()
    map_pipe = MapPipe(subpipe)
    
    assert map_pipe.apply([10, 20], {"offset": 5}) == [15, 25]
