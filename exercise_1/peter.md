# Double
| Implementierung      | Matrixgröße | Verifikation | Zeit [ms]  | Bemerkung               |
|----------------------|--------------|---------------|-------------|--------------------------|
| `matrix_mul`         | 512          | OK            | 512.229     | CPU (seriell)           |
| `matrix_mul`         | 1024         | OK            | 501.053     | CPU (seriell)           |
| `matrix_mul`         | 2000         | OK            | 506.130     | CPU (seriell)           |
| `matrix_mul`         | 2048         | OK            | 510.077     | CPU (seriell)           |
| `matrix_mul_omp`     | 512          | OK            | 133.054     | CPU mit OpenMP          |
| `matrix_mul_omp`     | 1024         | OK            | 122.730     | CPU mit OpenMP          |
| `matrix_mul_omp`     | 2000         | OK            | 122.838     | CPU mit OpenMP          |
| `matrix_mul_omp`     | 2048         | OK            | 118.870     | CPU mit OpenMP          |
| `matrix_mul_N`       | 512          | OK            | 201.040     | GPU (OpenCL Kernel)     |
| `matrix_mul_N`       | 1024         | OK            | 1587.323    | GPU (OpenCL Kernel)     |
| `matrix_mul_N`       | 2048         | OK            | 3097.458    | GPU (OpenCL Kernel)     |
| `matrix_mul_N`       | 2000         | OK            | 14772.399   | GPU (OpenCL Kernel)     |

# Float
| Implementierung      | Matrixgröße | Verifikation | Zeit [ms]  | Bemerkung               |
|----------------------|------------|--------------|------------|------------------------|
| `matrix_mul`         | 512        | OK           | 258.660    | CPU (seriell)          |
| `matrix_mul`         | 1024       | OK           | 262.820    | CPU (seriell)          |
| `matrix_mul`         | 2000       | OK           | 269.249    | CPU (seriell)          |
| `matrix_mul`         | 2048       | OK           | 262.089    | CPU (seriell)          |
| `matrix_mul_omp`     | 512        | OK           | 69.889     | CPU mit OpenMP         |
| `matrix_mul_omp`     | 1024       | OK           | 90.677     | CPU mit OpenMP         |
| `matrix_mul_omp`     | 2000       | OK           | 80.927     | CPU mit OpenMP         |
| `matrix_mul_omp`     | 2048       | OK           | 77.028     | CPU mit OpenMP         |
| `matrix_mul_N`       | 512        | OK           | 140.003    | GPU (OpenCL Kernel)    |
| `matrix_mul_N`       | 1024       | OK           | 346.558    | GPU (OpenCL Kernel)    |
| `matrix_mul_N`       | 2000       | OK           | 2139.194   | GPU (OpenCL Kernel)    |
| `matrix_mul_N`       | 2048       | OK           | 11871.306  | GPU (OpenCL Kernel)    |

