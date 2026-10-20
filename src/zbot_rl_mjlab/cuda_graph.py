"""Opt-in capture for deterministic fixed-shape CUDA tensor functions only."""

from collections.abc import Callable

import torch


class FixedTensorGraph:
    """Capture a pure tensor function; replay returns owned output storage.

    The callable must not mutate model state, sample random numbers, synchronize,
    or replace parameter storage. In-place parameter updates remain visible.
    Capture never changes module training mode. No automatic fallback is used:
    shape/device changes are errors, rather than silently stale graph inputs.
    """

    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor], example: torch.Tensor):
        if example.device.type != 'cuda':
            raise ValueError('FixedTensorGraph requires a CUDA tensor')
        self.function = function  # Keep captured modules and their parameter storage alive.
        self.input = example.detach().clone()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(example.device), torch.no_grad():
            stream = torch.cuda.Stream(device=example.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(self.input)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(self.graph):
                self.output = function(self.input)
        if not isinstance(self.output, torch.Tensor):
            raise TypeError('captured function must return one tensor')

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if (value.shape != self.input.shape or value.dtype != self.input.dtype
                or value.device != self.input.device):
            raise ValueError('CUDA graph input shape, dtype and device must match capture')
        with torch.cuda.device(value.device), torch.no_grad():
            self.input.copy_(value)
            self.graph.replay()
            return self.output.clone()
