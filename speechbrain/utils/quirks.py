"""Platform/GPU-specific quirks, i.e. workarounds and saner defaults due to
platform-specific issues.

Author:
    * Sylvain de Langen 2024
"""

import logging

import torch

logger = logging.getLogger(__name__)


def disable_cudnn_benchmarking():
    """Disables CuDNN benchmarking. no-op on platforms where it is already off
    by default.

    Benchmarking, when enabled, theoretically improves convolution performance
    by automatically comparing different kernels for some operations.

    However, benchmarking has to be re-run for every unique input shape, which
    makes it unsuitable for highly dynamic shapes.
    Since SpeechBrain does tend to use very varied shapes without attempting to
    pad the differences out, leaving benchmarking on can severely degrade
    training performance.

    This function disables it as we deem no-benchmarking to be a saner default
    to avoid performance bugs at the moment.

    As of PyTorch 2.3.0, the default is `False` for CUDA GPUs, but `True`
    for HIP GPUs.

    The HIP equivalent to CuDNN is MIOpen, but it is controlled through the same
    PyTorch API.
    """

    logger.info(
        "... Setting `torch.backends.cudnn.benchmark = False`. "
        "See `disable_cudnn_benchmarking` in `speechbrain/utils/quirks.py`."
    )
    torch.backends.cudnn.benchmark = False


def apply_hip_quirks():
    """Apply quirks specific to AMD HIP."""

    logger.info("Detected AMD HIP. Applying HIP-specific quirks...")

    disable_cudnn_benchmarking()


def apply_quirks():
    """Apply quirks depending on the platform."""

    # AMD HIP?
    if torch.cuda.is_available() and torch.version.hip:
        apply_hip_quirks()
