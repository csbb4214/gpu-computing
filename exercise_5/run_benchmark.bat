@echo off
rem Requires: make in PATH, run from repository folder containing the Makefile.

setlocal

rem --- config ---
set RESULTS=results.csv
set N_VALUES=1024 1048576 536870912
set RUNS=10

rem --- clean build log ---
if exist build.log del build.log

rem --- CSV header ---
echo version,precision,N,result,elapsed_ms > "%RESULTS%"

for %%N in (%N_VALUES%) do (
  rem -----------
  rem --- int ---
  rem -----------

  echo [INFO] Building int N=%%N

  call mingw32-make all N=%%N > build_temp.log 2>&1
  if errorlevel 1 (
    echo [ERROR] Build failed for int N=%%N
    type build_temp.log >> build.log
    del build_temp.log
    exit /b 1
  )

  rem Check for warnings
  findstr /i /c:"warning" build_temp.log >nul
  if not errorlevel 1 (
    echo [WARNING] Build warnings for int N=%%N
    type build_temp.log >> build.log
  )
  del build_temp.log

  echo [INFO] Running int binaries...

  for /l %%x in (1, 1, %RUNS%) do (
    call ".\sequential_reduction_N%%N.exe" >> "%RESULTS%"
    call ".\parallel_reduction_N%%N.exe" >> "%RESULTS%"
    call ".\multistage_reduction_N%%N.exe" >> "%RESULTS%"
  )

  rem -------------
  rem --- float ---
  rem -------------

  echo [INFO] Building float N=%%N IT=%%I

  call mingw32-make all N=%%N FLOAT=1 > build_temp.log 2>&1
  if errorlevel 1 (
    echo [ERROR] Build failed for float N=%%N
    type build_temp.log >> build.log
    del build_temp.log
    exit /b 1
  )

  rem Check for warnings
  findstr /i /c:"warning" build_temp.log >nul
  if not errorlevel 1 (
    echo [WARNING] Build warnings for double N=%%N
    type build_temp.log >> build.log
  )
  del build_temp.log

  echo [INFO] Running float binaries...

  for /l %%x in (1, 1, %RUNS%) do (
    call ".\sequential_reduction_N%%N_float.exe" >> "%RESULTS%"
    call ".\parallel_reduction_N%%N_float.exe" >> "%RESULTS%"
    call ".\multistage_reduction_N%%N_float.exe" >> "%RESULTS%"
  )
)

rem --- cleanup ---
timeout /t 5 > nul
call mingw32-make clean

echo [DONE] Results written to "%RESULTS%".
if exist build.log (
 echo [NOTE] Warnings/errors logged to build.log
)
endlocal
exit /b 0