@echo off
rem Batch script to build/run jacobi variants and collect CSV results.
rem Requires: make in PATH, run from repository folder containing the Makefile.

setlocal

rem --- config ---
set RESULTS=results.csv
set N_VALUES=1024 2048
set IT_VALUES=10 100 1000

rem --- CSV header ---
echo mode,precision,N,IT,time_ms > "%RESULTS%"

for %%N in (%N_VALUES%) do (
  for %%I in (%IT_VALUES%) do (

    rem --- double precision ---

    echo [INFO] Building double N=%%N IT=%%I 

    call mingw32-make all N=%%N IT=%%I > build.log 2>&1
    if errorlevel 1 (
      echo [ERROR] Build failed for double N=%%N IT=%%I
      exit /b 1
    )

    echo [INFO] Running double binaries...

    call ".\jacobi_N%%N_IT%%I.exe" >> "%RESULTS%"
    call ".\jacobi_omp_N%%N_IT%%I.exe" >> "%RESULTS%"
    call ".\jacobi_ocl_N%%N_IT%%I_V1.exe" >> "%RESULTS%"
    call ".\jacobi_ocl_N%%N_IT%%I_V2.exe" >> "%RESULTS%"


    rem --- float precision ---

    echo [INFO] Building float N=%%N IT=%%I

    call mingw32-make all N=%%N IT=%%I FLOAT=1 > build.log 2>&1
    if errorlevel 1 (
      echo [ERROR] Build failed for float N=%%N IT=%%I
      exit /b 1
    )

    echo [INFO] Running float binaries...

    call ".\jacobi_N%%N_IT%%I_float.exe" >> "%RESULTS%"
    call ".\jacobi_omp_N%%N_IT%%I_float.exe" >> "%RESULTS%"
    call ".\jacobi_ocl_N%%N_IT%%I_V1.exe" >> "%RESULTS%"
    call ".\jacobi_ocl_N%%N_IT%%I_V2.exe" >> "%RESULTS%"

  )
)

rem --- cleanup ---
timeout /t 2 > nul
call mingw32-make clean

echo [DONE] Results written to "%RESULTS%".
endlocal
exit /b 0