#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import scipy.io


def display_item_summary(key, item):
    if not hasattr(item, "items"):
        print(key)
        if hasattr(item, "shape"):
            print(f"    Shape: {item.shape}")
        if hasattr(item, "dtype"):
            print(f"    Dtype: {item.dtype}")
    else:
        for nested_key, nested_value in item.items():
            display_item_summary(nested_key, nested_value)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args(argv)

    display_item_summary(None, scipy.io.loadmat(args.filename))
    return 0
