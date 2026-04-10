#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from xommons.tools import convert_json_to_images_and_texts
from xommons.tools import dumpmat
from xommons.tools import gather_tiffs
from xommons.tools import npy2tiff
from xommons.tools import npz2tiff
from xommons.tools import plot_mat
from xommons.tools import showh5
from xommons.tools import showimg
from xommons.tools import showprobe
from xommons.tools import showptychodus
from xommons.tools import summarize_mat
from xommons.tools import unwrap_phase


COMMANDS = {
    "convert_json_to_images_and_texts": convert_json_to_images_and_texts.main,
    "dumpmat": dumpmat.main,
    "gather_tiffs": gather_tiffs.main,
    "npy2tiff": npy2tiff.main,
    "npz2tiff": npz2tiff.main,
    "plot_mat": plot_mat.main,
    "showh5": showh5.main,
    "showimg": showimg.main,
    "showprobe": showprobe.main,
    "showptychodus": showptychodus.main,
    "summarize_mat": summarize_mat.main,
    "unwrap_phase": unwrap_phase.main,
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="xommons")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, remaining = parser.parse_known_args(argv)
    return COMMANDS[args.command](remaining)
