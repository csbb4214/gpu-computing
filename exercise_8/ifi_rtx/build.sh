#!/usr/bin/env bash
set -euo pipefail

make clean || true

# Required int sizes
make all N=1024
make all N=1048576
make all N=536870912
