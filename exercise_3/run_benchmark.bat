@echo off
rem Batch script to build/run jacobi variants and collect CSV results.
rem Requires: make in PATH, run from repository folder containing the Makefile.

setlocal

rem --- config ---
set RESULTS=results.csv
set N_VALUES=1024 2048
set IT_VALUES=10 100 1000

rem --- clean build log ---
if exist build.log del build.log

rem --- clean detail files ---
del kernel_times_*.csv 2>nul

rem --- CSV header ---
echo mode,precision,N,IT,total_write,total_kernel,total_read,write_f,write_tmp,write_u > "%RESULTS%"

for %%N in (%N_VALUES%) do (
  for %%I in (%IT_VALUES%) do (

    rem ------------------------
    rem --- double precision ---
    rem ------------------------

    echo [INFO] Building double N=%%N IT=%%I 

    call mingw32-make all N=%%N IT=%%I > build_temp.log 2>&1
    if errorlevel 1 (
      echo [ERROR] Build failed for double N=%%N IT=%%I
      type build_temp.log >> build.log
      del build_temp.log
      exit /b 1
    )

    rem Check for warnings
    findstr /i /c:"warning" build_temp.log >nul
    if not errorlevel 1 (
      echo [WARNING] Build warnings for double N=%%N IT=%%I
      type build_temp.log >> build.log
    )
    del build_temp.log

    echo [INFO] Running double binaries...

    call ".\jacobi_ocl_N%%N_IT%%I_V2.exe" >> "%RESULTS%"

    rem -----------------------
    rem --- float precision ---
    rem -----------------------

    echo [INFO] Building float N=%%N IT=%%I

    call mingw32-make all N=%%N IT=%%I FLOAT=1 > build_temp.log 2>&1
    if errorlevel 1 (
      echo [ERROR] Build failed for float N=%%N IT=%%I
      type build_temp.log >> build.log
      del build_temp.log
      exit /b 1
    )

    rem Check for warnings
    findstr /i /c:"warning" build_temp.log >nul
    if not errorlevel 1 (
      echo [WARNING] Build warnings for double N=%%N IT=%%I
      type build_temp.log >> build.log
    )
    del build_temp.log

    echo [INFO] Running float binaries...

    call ".\jacobi_ocl_N%%N_IT%%I_float_V2.exe" >> "%RESULTS%"

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