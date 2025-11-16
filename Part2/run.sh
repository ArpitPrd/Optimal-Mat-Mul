#!/bin/bash

# Usage:
#   ./run.sh baseline 1000
#   ./run.sh optimized 2000

if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh <baseline|optimized> <integer>"
    exit 1
fi

VERSION=$1
ARG=$2

if [ "$VERSION" = "baseline" ]; then
    BIN="./baseline/bin/sw_baseline"
elif [ "$VERSION" = "optimized" ]; then
    BIN="./optimized/bin/sw_opt"
else
    echo "Invalid version: $VERSION"
    echo "Use: baseline OR optimized"
    exit 1
fi

if [ ! -f "$BIN" ]; then
    echo "Executable not found: $BIN"
    echo "Run: make"
    exit 1
fi

echo "Running $VERSION with argument $ARG ..."
$BIN "$ARG"
