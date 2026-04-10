#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import numpy as np
import tifffile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    parser.add_argument("--format", default="2idd")
    args = parser.parse_args(argv)

    with open(args.fname, "r", encoding="utf-8") as f:
        json_table = json.load(f)

    if args.format != "2idd":
        raise ValueError(f"Unsupported format: {args.format}")

    meta_dict = {}
    for key, data in json_table.items():
        if isinstance(data, list):
            tifffile.imwrite(f"{key}.tiff", np.array(data))
        else:
            meta_dict[key] = data

    with open("metadata.json", "w", encoding="utf-8") as f_meta:
        json.dump(meta_dict, f_meta)

    return 0
