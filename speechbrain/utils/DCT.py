"""Configuration and utility classes for classes for Dynamic Chunk Training
Authors
* Sylvain de Langen 2023
"""

from speechbrain.core import Stage
from dataclasses import dataclass
from typing import Optional

import torch

# NOTE: this configuration object is intended to be relatively specific to DCT;
# if you want to implement a different similar type of chunking different from
# DCT you should consider using a different object.
@dataclass
class DCTConfig:
    """Dynamic Chunk Training configuration object for use with transformers,
    often in ASR for streaming.

    This object may be used both to configure masking at training time and for
    run-time configuration of DCT-ready models."""

    chunk_size: int
    """Size in frames of a single chunk, always `>0`.
    If chunkwise streaming should be disabled at some point, pass an optional
    streaming config parameter."""

    left_context_size: Optional[int]
    """Number of *chunks* (not frames) visible to the left, always `>=0`.
    If zero, then chunks can never attend to any past chunk.
    If `None`, the left context is infinite (but use
    `.is_fininite_left_context` for such a check)."""

    def is_infinite_left_context(self) -> bool:
        """Returns true if the left context is infinite (i.e. any chunk can
        attend to any past frame)."""
        return self.left_context_size is not None

# TODO
@dataclass
class DCTConfigRandomSampler:
    dct_prob: float

    chunk_size_min: int
    chunk_size_max: int

    limited_left_context_prob: float
    left_context_chunks_min: int
    left_context_chunks_max: int

    test_config: Optional[DCTConfig] = None
    valid_config: Optional[DCTConfig] = None

    def _sample_bool(prob: float):
        return torch.rand((1,)).item() < prob

    def __call__(self, stage: Stage) -> DCTConfig:
        if stage == Stage.TRAIN:
            # When training for streaming, for each batch, we have a
            # `dynamic_chunk_prob` probability of sampling a chunk size
            # between `dynamic_chunk_min` and `_max`, otherwise output
            # frames can see anywhere in the future.
            # NOTE: We use torch random to be bound to the experiment seed.
            if self._sample_bool(self.dct_prob):
                chunk_size = torch.randint(
                    self.chunk_size_min,
                    self.chunk_size_max + 1,
                    (1,),
                ).item()

                if self._sample_bool(self.limited_left_context_prob):
                    left_context_chunks = torch.randint(
                        self.left_context_chunks_max,
                        self.left_context_chunks_max + 1,
                        (1,),
                    ).item()
                else:
                    left_context_chunks = None
                
                return DCTConfig(chunk_size, left_context_chunks)
            return None
        elif stage == Stage.TEST:
            return self.test_config
        elif stage == Stage.VALID:
            return self.valid_config
        else:
            raise AttributeError(f"Unsupported stage found {stage}")

