#!/bin/bash
echo "Building OpenCL program..."
gcc -o "${1%.*}.exe" "$1" -lOpenCL -DCL_TARGET_OPENCL_VERSION=220 -Wall

if [ $? -eq 0 ]; then
    echo "Build successful! Running program..."
    ./"${1%.*}.exe"
else
    echo "Build failed!"
fi