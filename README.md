*Xommons* is a collection of convenient tools for synchrotron data examination, assessment and analysis. It offers a range of utility functions that can be integrated into your workflow as well as a toolbox of macros that can be run from terminal as if they are bash commands. For example:

```bash
showptychodus ptychodus_para.hdf5 probe
```
will display the probes in a Ptychodus parameter HDF5 file.

## Installation

First clone this repository.

### Option 1
- `pip install .`

### Option 2
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
- In the project root directory, do `uv venv`
- `uv sync`

## Post-installation

Run `./setup.sh` from the repository root to append the `macros` folder to your `PATH`. Restart your shell (or source `~/.bashrc`) before invoking the macros directly from the terminal.
