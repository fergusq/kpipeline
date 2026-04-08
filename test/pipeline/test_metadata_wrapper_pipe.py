from kpipeline.pipeline import MetadataWrapperPipe
from test.pipeline.common import SimplePipe


def test_metadata_wrapper_pipe():
    def to_inner(m): return {"offset": m["value"] * 2}
    subpipe = SimplePipe()
    wrapper = MetadataWrapperPipe(to_inner, subpipe)
    
    # metadata {"value": 5} -> inner {"offset": 10} -> 10 + 10 = 20
    assert wrapper.apply(10, {"value": 5}) == 20
