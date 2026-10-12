# kpipeline

`kpipeline` is a simple pipeline library that allows you to write your complex application as a pipeline. This helps code organization, debugging, and testing.

## How to use

kpipe is like a DSL language inside Python. It is based on "pipes" which can be combined into pipelines using the pipe primitives provided by the library.

Each pipe represents a function transforming an input into an output. Pipes are stateless and immutable (with the exception of certain async only pipes). Below is an example defining two example pipes.

```python
from kpipeline import Pipe

class AddOnePipe(Pipe[int, int, None]):
    def apply(self, input: int, metadata: None) -> int:
        return input + 1


class MulByTwoPipe(Pipe[int, int, None]):
    def apply(self, input: int, metadata: None) -> int:
        return input * 2
```

These pipes can be combined using the ChainPipe primitive (aliased into the `|` operator) two form a pipeline that performs these two pipes sequentially:

```python
pipeline = AddOnePipe() | MulByTwoPipe()

# or

from kpipeline import ChainPipe
pipeline = ChainPipe(AddOnePipe(), MulByTwoPipe())

print(pipeline.apply(2))  # 6
```

Currently, the library defines these primitives:

| Primitive | Purpose |
|---|---|
|`ChainPipe`|Execute two pipes sequentially|
|`ConditionalPipe`|Execute a pipe if a given condition is true|
|`BranchPipe`|Execute one of two pipes depending on whether the condition is true or false|
|`SelectPipe`|Execute one of multiple pipes based on a selector key|
|`ParallelPipe`|Execute multiple pipes and combine their results (only async implementation is actually parallel)|
|`MetadataWrapperPipe`|Transform the metadata given to the pipeline for the subpipe|
|`WrapperPipe`|Wrap a pipe so that its input is extracted from the outer input and its output is combined with the outer input|
|`MapPipe`|Apply a subpipe into a sequence of inputs|
|`FilterPipe`|Filter a sequence of inputs using a predicate|
|`RetryPipe`|Run a pipe multiple times in case it fails|
|`FallbackPipe`|Runs a fallback pipe in case the subpipe fails|
|`AsyncBatchCollectorPipe`|Waits for a specific number of seconds, accumulating incoming data, and then processes them as a batch (async only, stateful)|
|`AsyncSemaphorePipe`|Limits the number of concurrent calls to the subpipe (async only, stateful)|
|`AsyncTimeoutPipe`|Cancels the subpipe after a certain amount of time (async only)|

## Tests

You can run all tests in this repository with

```
uv run pytest test
uv run mypy .
```

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

## AI Use Disclosure

I dislike writing tests so unit tests have been generated for your convenience.
They are provided in the hope that they are better than nothing.
