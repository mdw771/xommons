#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import tifffile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--type", choices=["magnitude", "phase"], default="magnitude")
    args = parser.parse_args(argv)

    arr = np.load(args.file)
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

    base = os.path.splitext(os.path.basename(args.file))[0]
    if args.type == "magnitude":
        tifffile.imwrite(f"{base}_mag.tiff", np.abs(arr))
    else:
        tifffile.imwrite(f"{base}_phase.tiff", np.angle(arr))

    return 0
