from kpipeline.pipeline import IdentityPipe


def test_identity_pipe():
    pipe = IdentityPipe[int, dict]()
    assert pipe.apply(10, {}) == 10
    assert pipe.batch_apply([10, 20], {}) == [10, 20]
