serial (double):

$ ./matrix_mul_N512.exe
Verification:   OK
Time:   152.000 ms

$ ./matrix_mul_N1024.exe
Verification:   OK
Time:  1624.000 ms

$ ./matrix_mul_N2048.exe
Verification:   OK
Time: 22097.000 ms

$ ./matrix_mul_N2000.exe
Verification:   OK
Time:  6754.000 ms

serial (float):

$ ./matrix_mul_N512.exe
Verification:   OK
Time:    37.000 ms

$ ./matrix_mul_N1024.exe 
Verification:   OK
Time:   786.000 ms

$ ./matrix_mul_N2048.exe 
Verification:   OK
Time:  9179.000 ms

$ ./matrix_mul_N2000.exe 
Verification:   OK
Time:  2406.000 ms



openMP (double):

$ ./matrix_mul_omp_N512.exe 
Verification:   OK
Time:    20.000 ms

$ ./matrix_mul_omp_N1024.exe
Verification:   OK
Time:   213.000 ms

$ ./matrix_mul_omp_N2048.exe
Verification:   OK
Time:  1909.000 ms

$ ./matrix_mul_omp_N2000.exe
Verification:   OK
Time:  1627.000 ms

openMP (float):

$ ./matrix_mul_omp_N512.exe
Verification:   OK
Time:     5.000 ms

$ ./matrix_mul_omp_N1024.exe
Verification:   OK
Time:    86.000 ms

$ ./matrix_mul_omp_N2048.exe 
Verification:   OK
Time:   825.000 ms

$ ./matrix_mul_omp_N2000.exe 
Verification:   OK
Time:   697.000 ms



Radeon RX 6700XT (double):

$ ./matrix_mul_N512.exe 
Verification:   OK
Time:    15.000 ms

$ ./matrix_mul_N1024.exe
Verification:   OK
Time:   105.000 ms

$ ./matrix_mul_N2048.exe
Verification:   OK
Time:  1019.000 ms

$ ./matrix_mul_N2000.exe
Verification:   OK
Time:   404.000 ms

Radeon RX 6700XT (float):

$ ./matrix_mul_N512.exe 
Verification:   OK
Time:    15.000 ms

$ ./matrix_mul_N1024.exe
Verification:   OK
Time:   106.000 ms

$ ./matrix_mul_N2048.exe
Verification:   OK
Time:   983.000 ms

$ ./matrix_mul_N2000.exe
Verification:   OK
Time:   381.000 ms