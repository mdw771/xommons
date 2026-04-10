#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tifffile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--queries", nargs="+", default=[])
    args = parser.parse_args(argv)

    try:
        f = h5py.File(args.filename, "r")
        obj = f["object"]["real"][...] + 1j * f["object"]["imag"][...]
        obj = np.transpose(obj)
        shrink_len = (20 / 1200 * np.array(obj.shape)).astype(int)
        obj = obj[shrink_len[0] : -shrink_len[0], shrink_len[1] : -shrink_len[1]]
    except Exception:
        f = scipy.io.loadmat(args.filename)
        obj = f["object"]

    plt.figure()
    im = plt.imshow(np.angle(obj))
    plt.colorbar(im)
    plt.show()

    if args.save:
        tifffile.imwrite(f"{os.path.splitext(args.filename)[0]}.tiff", np.angle(obj))

    for query in args.queries:
        print(query)
        value = f
        for level in query.split("/"):
            value = value[level]
        print(value[...])

    return 0
