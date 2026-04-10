#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile

try:
    import ptychi.image_proc as ip
except ImportError:
    ip = None


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    if ip is None:
        raise ImportError("unwrap_phase requires Pty-Chi. Install the project with its declared uv dependencies.")

    arr = torch.tensor(np.load(args.filename))
    if arr.ndim == 2:
        arr = arr[None, ...]

    res = []
    for i_slice in range(arr.shape[0]):
        phase = ip.unwrap_phase_2d(
            arr[i_slice, ...],
            image_grad_method="fourier_differentiation",
            image_integration_method="fourier",
        )
        res.append(phase)

    res = torch.stack(res)
    if args.show:
        for i_slice in range(res.shape[0]):
            plt.imshow(res[i_slice].cpu().numpy())
            plt.colorbar()
            plt.show()

    tifffile.imwrite(f"{os.path.splitext(args.filename)[0]}_unwrapped_phase.tiff", res.cpu().numpy())
    return 0
