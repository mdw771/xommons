*Xommons* is a collection of convenient tools for synchrotron data examination, assessment and analysis. It offers a range of utility functions that can be integrated into your workflow as well as a uv-managed command-line toolbox. For example:

```bash
xommons showptychodus ptychodus_para.hdf5 probe
```
will display the probes in a Ptychodus parameter HDF5 file.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install the package from PyPI as a tool:

```bash
uv tool install xommons
```

## Usage

If `uv` tells you its tool bin directory is not on `PATH`, run:

```bash
uv tool update-shell
```

After that, use the single launcher with subcommands:

```bash
xommons showh5 somefile.hdf5
xommons showprobe probe.npy
```

The same installed tool can be invoked through `uvx`:

```bash
uvx xommons showh5 somefile.hdf5
```

`unwrap_phase` has heavier optional dependencies. If you need that subcommand, install the tool with the `unwrap-phase` extra:

```bash
uv tool install 'xommons[unwrap-phase]'
uvx xommons unwrap_phase phase.npy
```
