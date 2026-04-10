#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as plt
import numpy as np

import xommons.plot as xp
from xommons.tools._image import load_image_file


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    parser.add_argument("--component", choices=["real", "imag", "mag", "phase"], default="mag")
    args = parser.parse_args(argv)

    img = load_image_file(args.fname)
    if img.ndim == 2:
        img = img[None, None, ...]
    if img.ndim == 3:
        img = img[None, ...]

    if args.component == "real":
        img = np.real(img)
    elif args.component == "imag":
        img = np.imag(img)
    elif args.component == "mag":
        img = np.abs(img)
    else:
        img = np.angle(img)

    xp.plot_ptychi_probe(img)
    plt.show()
    return 0
