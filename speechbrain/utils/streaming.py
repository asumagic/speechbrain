"""Utilities to assist with designing and training streaming models.

Authors
* Sylvain de Langen 2023
"""

import math
import torch
from typing import Callable


def chunkify_sequence(x, chunk_size):
    chunks = []

    for i in range(max(1, math.ceil(x.shape[1] / chunk_size))):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(x[:,start:end,...])

    return chunks


def merge_chunks(chunks):
    return torch.cat(chunks, 1)


def chunked_wav_lens(chunks, wav_lens):
    chunk_wav_lens = []

    # consider 3 chunks: we have chunk_frac at 0.0, 0.333, 0.666
    # for value 0.7:
    # - the first two chunks are trivially 1.0.
    # - the last chunk is (value - 0.666) / (1 / chunks)

    for i in range(len(chunks)):
        chunk_frac = i / len(chunks)
        chunk_raw_len = (wav_lens - chunk_frac) * len(chunks)
        chunk_raw_len = torch.clamp(chunk_raw_len, 0.0, 1.0)
        chunk_wav_lens.append(chunk_raw_len)

    return chunk_wav_lens


def infer_dependency_matrix(
    model: Callable,
    seq_shape: tuple,
    in_stride: int = 1
):
    """
    Randomizes parts of the input sequence several times in order to detect
    dependencies between input frames and output frames, aka whether a given
    output frame depends on a given input frame.

    This can prove useful to check whether a model behaves correctly in a
    streaming context and does not contain accidental dependencies to future
    frames that couldn't be known in a streaming scenario.

    Note that this can get very computationally expensive for very long
    sequences.

    Furthermore, this expects inference to be fully deterministic, else false
    dependencies may be found. This also means that the model must be in eval
    mode, to inhibit things like dropout layers.

    Arguments
    ---------
    model : Callable
        Can be a model or a function (potentially emulating streaming
        functionality). Does not require to be a trained model, random weights
        should usually suffice.
    seq_shape : tuple
        The function tries inferring by randomizing parts of the input sequence
        in order to detect unwanted dependencies.
        The shape is expected to look like `[batch_size, seq_len, num_feats]`,
        where `batch_size` may be `1`.
    in_stride : int
        Consider only N-th input, for when the input sequences are very long
        (e.g. raw audio) and the output is shorter (subsampled, filters, etc.)

    Returns
    -------
    dependencies : torch.BoolTensor
        Matrix representing whether an output is dependent on an input; index
        using `[in_frame_idx, out_frame_idx]`. `True` indicates a detected
        dependency.
    """
    # TODO: document arguments

    bs, seq_len, feat_len = seq_shape

    base_seq = torch.rand(seq_shape)
    with torch.no_grad():
        base_out = model(base_seq)

        if not model(base_seq).equal(base_out):
            raise ValueError(
                "Expected deterministic model, but inferring twice on the same "
                "data yielded different results. Make sure that you use "
                "`eval()` mode so that it does not include randomness."
            )
    out_len, _out_feat_len = base_out.shape[1:]

    deps = torch.zeros(((seq_len + (in_stride - 1)) // in_stride, out_len), dtype=torch.bool)

    for in_frame_idx in range(0, seq_len, in_stride):
        test_seq = base_seq.clone()
        test_seq[:,in_frame_idx,:] = torch.rand(bs, feat_len)

        with torch.no_grad():
            test_out = model(test_seq)

        for out_frame_idx in range(out_len):
            if not torch.allclose(
                test_out[:,out_frame_idx,:],
                base_out[:,out_frame_idx,:]
            ):
                deps[in_frame_idx // in_stride][out_frame_idx] = True

    return deps

def plot_dependency_matrix(deps, in_stride: int = 1):
    """
    Returns a matplotlib figure of a dependency matrix generated by
    `infer_dependency_matrix`.

    At a given point, a red square indicates that a given output frame (y-axis)
    was to depend on a given input frame (x-axis).

    For example, a fully red image means that all output frames were dependent
    on all the history. This could be the case of a bidirectional RNN, or a
    transformer model, for example.

    Arguments
    ---------
    deps : torch.BoolTensor
        Matrix returned by `infer_dependency_matrix` or one in a compatible
        format.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "red"])

    fig, ax = plt.subplots()
    
    ax.pcolormesh(
        torch.permute(deps, (1, 0)),
        cmap=cmap,
        vmin=False,
        vmax=True,
        edgecolors="gray",
        linewidth=0.5
    )
    ax.set_title("Dependency plot")
    ax.set_xlabel("in")
    ax.set_ylabel("out")
    ax.set_aspect("equal")
    return fig