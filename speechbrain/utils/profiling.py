"""Wrapper to handle PyTorch profiling and benchmarking.

Author:
    * Titouan Parcollet 2024
"""

import torch
import os


def default_trace_handler(prof, logdir):
    prof.export_stacks(os.path.join(logdir, "cpu_stacks.txt"), metric="self_cpu_time_total")
    prof.export_stacks(os.path.join(logdir, "test_stacks.txt"), metric="self_cuda_time_total")


def prepare_profiler(
    profile_warmup=5, profile_steps=5, logdir="tensorboard_logs"
):
    """Wrapper to create a PyTorch profiler to benchmark training of speechbrain.core.Brain instances.
    See ``torch.profiler.profile`` documentation for details (brief summary below).

    Arguments
    ---------
    profile_warmup: int
        Number of warmup step before starting to log.
    profile_steps: int
        Number of steps to log after warmup.
    logdir: str
        Path to the output folder of the logs.

    Returns
    -------
    profiler
    """
    logdir = os.path.join(logdir, "profiler_logs")

    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=0, warmup=profile_warmup, active=profile_steps, repeat=1
        ),
        on_trace_ready=lambda prof: default_trace_handler(prof, logdir),
        # record_shapes=True,
        with_stack=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    )
