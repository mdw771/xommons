#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

import xommons.plot as xp


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    parser.add_argument("var", choices=["object", "probe", "positions", "pixelsize"])
    args = parser.parse_args(argv)

    with h5py.File(args.fname, "r") as f:
        if args.var == "object":
            fig = xp.plot_ptychi_object(f["object"][...])
            plt.show()
            plt.close(fig)
        elif args.var == "probe":
            fig = xp.plot_ptychi_probe(np.abs(f["probe"][...]))
            plt.show()
            plt.close(fig)
        elif args.var == "positions":
            positions = np.stack([f["probe_position_y_m"][...], f["probe_position_x_m"][...]], axis=-1)
            fig = xp.plot_ptychi_positions(positions)
            plt.show()
            plt.close(fig)
        else:
            pixel_height = f["object"].attrs["pixel_height_m"]
            pixel_width = f["object"].attrs["pixel_width_m"]
            print(f"Pixel height: {pixel_height} m, pixel width: {pixel_width} m")

    return 0
