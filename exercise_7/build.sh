#!/usr/bin/env bash
# Build all auto_levels binaries.
# Run on login node or locally in the repo with the Makefile.

set -euo pipefail

# --- clean build log ---
[ -f build.log ] && rm build.log

echo "[INFO] Building binaries"

if ! make all > build_temp.log 2>&1; then
  echo "[ERROR] Build failed"
  cat build_temp.log >> build.log
  rm -f build_temp.log
  exit 1
fi

# Check for warnings
if grep -qi "warning" build_temp.log; then
  echo "[WARNING] Build warnings"
  cat build_temp.log >> build.log
fi
rm -f build_temp.log

echo
echo "[DONE] All auto_levels binaries built"
if [ -f build.log ]; then
  echo "[NOTE] Warnings/errors logged to build.log"
fi

exit 0
