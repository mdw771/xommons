#!/usr/bin/env bash
set -euo pipefail

BASHRC_PATH="$HOME/.bashrc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACROS_PATH="$SCRIPT_DIR/macros"

touch "$BASHRC_PATH"

if ! grep -Fq "$MACROS_PATH" "$BASHRC_PATH"; then
    echo "Adding macros to .bashrc"
    printf '\nexport PATH="$PATH:%s"\n' "$MACROS_PATH" >> "$BASHRC_PATH"
else
    echo "Macros path already present in .bashrc"
fi
