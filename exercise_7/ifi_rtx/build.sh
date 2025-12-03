#!/bin/bash

set -euo pipefail

echo "[INFO] Building auto_levels binaries..."
echo "[INFO] Start time: $(date)"
echo

# alte build-Logs löschen
[ -f build.log ] && rm build.log

# build ausführen
if ! make all >build_temp.log 2>&1; then
  echo "[ERROR] Build failed"
  cat build_temp.log >>build.log
  rm -f build_temp.log
  exit 1
fi

# warnings prüfen
if grep -qi "warning" build_temp.log; then
  echo "[WARNING] Build warnings detected, see build.log"
  cat build_temp.log >>build.log
else
  echo "[INFO] Build finished without warnings"
fi

rm -f build_temp.log

echo "[INFO] Optionally cleaning with 'make clean' afterwards if you want."
echo "[INFO] End time: $(date)"
