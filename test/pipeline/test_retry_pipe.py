import pytest

from kpipeline.pipeline import Pipe, RetryPipe


def test_retry_pipe():
    class FailingPipe(Pipe[int, int, dict]):
        def __init__(self):
            self.count = 0
        def apply(self, data, metadata):
            self.count += 1
            if self.count < 3:
                raise ValueError("Fail")
            return data
            
    failing = FailingPipe()
    retry = RetryPipe(failing, retries=3, exceptions=ValueError)
    
    assert retry.apply(10, {}) == 10
    assert failing.count == 3

def test_retry_pipe_exhausted():
    class AlwaysFailingPipe(Pipe[int, int, dict]):
        def apply(self, data, metadata):
            raise ValueError("Fail")
            
    retry = RetryPipe(AlwaysFailingPipe(), retries=2, exceptions=ValueError)
    
    with pytest.raises(ValueError):
        retry.apply(10, {})
