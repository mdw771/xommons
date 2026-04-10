#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import os

import numpy as np
import tifffile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--pattern", default="*.tif*")
    parser.add_argument("--output", default="combined.tiff")
    args = parser.parse_args(argv)

    image_fnames = []
    for root, _, files in os.walk(args.dir):
        for file in files:
            relpath = os.path.join(root, file)
            if os.path.basename(relpath).startswith("."):
                continue
            if fnmatch.fnmatch(relpath, args.pattern):
                image_fnames.append(relpath)
    image_fnames.sort()
    print(image_fnames)

    images = [tifffile.imread(fname) for fname in image_fnames]
    if not images:
        raise ValueError("No matching image found.")

    if images[0].ndim == 2:
        output = np.array(images)
    else:
        output = np.concatenate(images)

    tifffile.imwrite(os.path.join(args.dir, args.output), output)
    return 0
