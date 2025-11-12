@echo off
rem Batch script to build/run jacobi variants and collect CSV results.
rem Requires: make in PATH, run from repository folder containing the Makefile.

setlocal

rem --- config ---
set RESULTS=results.csv
set N_VALUES=2048 4096
set IT_VALUES=10 100 1000
set RUNS=5

rem --- clean build log ---
if exist build.log del build.log

rem --- clean detail files ---
del kernel_times_*.csv 2>nul

rem --- CSV header ---
echo version,precision,N,IT,LOCAL_WORKGROUP_DIM_1,LOCAL_WORKGROUP_DIM_2,elapsed_ms > "%RESULTS%"

for %%N in (%N_VALUES%) do (
  for %%I in (%IT_VALUES%) do (

    rem ------------------------
    rem --- double precision ---
    rem ------------------------

    echo [INFO] Building double N=%%N IT=%%I 

    call mingw32-make all N=%%N IT=%%I LOCAL_WORKGROUP_DIM_1=16 LOCAL_WORKGROUP_DIM_2=16 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I LOCAL_WORKGROUP_DIM_1=8 LOCAL_WORKGROUP_DIM_2=32 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I LOCAL_WORKGROUP_DIM_1=4 LOCAL_WORKGROUP_DIM_2=64 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I LOCAL_WORKGROUP_DIM_1=64 LOCAL_WORKGROUP_DIM_2=4 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I LOCAL_WORKGROUP_DIM_1=2 LOCAL_WORKGROUP_DIM_2=128 > build_temp.log 2>&1
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

    for /l %%x in (1, 1, %RUNS%) do (
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-16_DIM2-16_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-16_DIM2-16_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-8_DIM2-32_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-8_DIM2-32_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-4_DIM2-64_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-4_DIM2-64_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-64_DIM2-4_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-64_DIM2-4_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-2_DIM2-128_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-2_DIM2-128_V3.exe" >> "%RESULTS%"
    )

    call mingw32-make clean

    rem -----------------------
    rem --- float precision ---
    rem -----------------------

    echo [INFO] Building float N=%%N IT=%%I

    call mingw32-make all N=%%N IT=%%I FLOAT=1 LOCAL_WORKGROUP_DIM_1=16 LOCAL_WORKGROUP_DIM_2=16 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I FLOAT=1 LOCAL_WORKGROUP_DIM_1=8 LOCAL_WORKGROUP_DIM_2=32 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I FLOAT=1 LOCAL_WORKGROUP_DIM_1=4 LOCAL_WORKGROUP_DIM_2=64 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I FLOAT=1 LOCAL_WORKGROUP_DIM_1=64 LOCAL_WORKGROUP_DIM_2=4 > build_temp.log 2>&1
    call mingw32-make all N=%%N IT=%%I FLOAT=1 LOCAL_WORKGROUP_DIM_1=2 LOCAL_WORKGROUP_DIM_2=128 > build_temp.log 2>&1
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

    for /l %%x in (1, 1, %RUNS%) do (
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-16_DIM2-16_float_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-16_DIM2-16_float_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-8_DIM2-32_float_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-8_DIM2-32_float_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-4_DIM2-64_float_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-4_DIM2-64_float_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-64_DIM2-4_float_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-64_DIM2-4_float_V3.exe" >> "%RESULTS%"

      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-2_DIM2-128_float_V2.exe" >> "%RESULTS%"
      call ".\jacobi_ocl_N%%N_IT%%I_DIM1-2_DIM2-128_float_V3.exe" >> "%RESULTS%"
    )

    call mingw32-make clean
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