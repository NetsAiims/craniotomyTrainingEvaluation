#!/bin/bash

echo "Starting Drilling Evaluation System..."

source ~/anaconda3/etc/profile.d/conda.sh
conda activate drilling

# Resolve the directory of this script, even if run via a symlink
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

sudo python GUI.py

echo "Done!!!"
