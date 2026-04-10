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
    parser.add_argument("var", choices=["object", "probe", "positions", "pixelsize", "ft-probe", "dp"])
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
        elif args.var == "pixelsize":
            pixel_height = f["object"].attrs["pixel_height_m"]
            pixel_width = f["object"].attrs["pixel_width_m"]
            print(f"Pixel height: {pixel_height} m, pixel width: {pixel_width} m")
        elif args.var == "ft-probe":
            probe = f["probe"][...]
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe[0, 0])) ** 2))
            ax[1].imshow((np.abs(np.fft.fftshift(np.fft.fft2(probe[0]), axes=(1, 2)) ** 2)).sum(axis=0))
            ax[0].set_title("First mode")
            ax[1].set_title("All modes")
            plt.show()
            plt.close(fig)
        elif args.var == "dp":
            if "dp" not in f:
                raise ValueError("Dataset 'dp' not found in file. Make sure your input file is a DP file.")
            dp = f["dp"][0]
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(dp)
            plt.colorbar(im)
            plt.show()
            plt.close(fig)

    return 0
