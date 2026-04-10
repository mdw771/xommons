#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as plt
import numpy as np

from xommons.tools._image import load_image_file


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    args = parser.parse_args(argv)

    img = load_image_file(args.fname)
    if img.ndim == 2:
        img = img[None, ...]

    ncols = 2 if np.iscomplexobj(img) else 1
    _, ax = plt.subplots(len(img), ncols, squeeze=False)
    for i, im in enumerate(img):
        if np.iscomplexobj(img):
            ax[i, 0].imshow(np.abs(im))
            ax[i, 0].set_title("magnitude")
            ax[i, 1].imshow(np.angle(im))
            ax[i, 1].set_title("phase")
        else:
            ax[i, 0].imshow(im)

    plt.show()
    return 0
