#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import h5py


def print_h5_group_recursive(grp: h5py.Group | h5py.File, indent: int = 0):
    indent_string = "  " * indent
    for key in grp.keys():
        if isinstance(grp[key], h5py.Group):
            print(f"{indent_string}Group: {key}")
            print_h5_group_recursive(grp[key], indent + 1)
        else:
            if grp[key].size < 2:
                print(f"{indent_string}Dataset: {key} = {grp[key][...]}")
            else:
                print(f"{indent_string}Dataset: {key}: {grp[key]}")
            if len(grp[key].attrs) > 0:
                print(f"{indent_string}  Attributes:")
                for attr_name, attr_value in grp[key].attrs.items():
                    print(f"{indent_string}    {attr_name}: {attr_value}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args(argv)

    with h5py.File(args.filename, "r") as f:
        print_h5_group_recursive(f)

    return 0
