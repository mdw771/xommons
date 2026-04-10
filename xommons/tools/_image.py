#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import tifffile


def load_image_file(fname: str):
    ext = os.path.splitext(fname)[-1].lower()
    if ext in {".tif", ".tiff"}:
        return tifffile.imread(fname)
    if ext == ".npy":
        return np.load(fname)
    raise ValueError(f"Unknown extension: {ext}")
