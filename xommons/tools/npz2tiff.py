#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import tifffile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args(argv)

    archive = np.load(args.file)
    print(f"Here are the keys in {args.file}. Which one do you want to open?")
    print(list(archive.keys()))
    key = input().strip()
    dset = archive[key]
    print(
        "The shape of the selected array is {}. Enter the indices for the slice that you want to dump. "
        "Separate axes by comma and start/end within an axis by colon. To slice an entire axis, leave it "
        "blank or just put a colon.".format(dset.shape)
    )

    slicers = []
    slicer_input = input().strip()
    for slicer in slicer_input.split(","):
        slicer = slicer.strip()
        if len(slicer) == 0 or slicer == ":":
            slicers.append(slice(None))
        elif ":" not in slicer:
            slicers.append(int(slicer))
        else:
            start_text, end_text = slicer.split(":", 1)
            start = int(start_text) if start_text else None
            end = int(end_text) if end_text else None
            slicers.append(slice(start, end))

    data = dset[tuple(slicers)]
    if np.iscomplexobj(data):
        complex_input = input(
            f"The values are {data.dtype}. Do you want the (1) magnitude or (2) phase? "
        ).strip()
        data = np.abs(data) if complex_input == "1" else np.angle(data)

    print(f"Writing an array of shape {data.shape}.")
    tifffile.imwrite(f"{os.path.splitext(args.file)[0]}.tiff", data)
    return 0
