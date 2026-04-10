#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import xommons.io as xio


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--key", default="object", help="key to dump")
    args = parser.parse_args(argv)

    try:
        f = xio.load_h5(args.filename)
    except Exception:
        f = xio.load_mat(args.filename)

    try:
        val = f[args.key]
        if args.key in ["object", "probe"]:
            val = xio.matlab_complex_to_array(val)
            xio.complex_image_to_tiff(val, os.path.splitext(args.filename)[0])
        else:
            print(val)
    finally:
        if hasattr(f, "close"):
            f.close()

    return 0
