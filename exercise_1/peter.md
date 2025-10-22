# Lokal

## Double
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

## Float
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


# Cluster

## Double
| Implementierung | Matrixgröße | Verifikation | Zeit [ms] | Bemerkung                  |
|-----------------|-------------|--------------|-----------|----------------------------|
| matrix_mul          | 512        | OK           | 673.174   | Seriell        |
| matrix_mul          | 1024       | OK           | 700.706   | Seriell        |
| matrix_mul          | 2000       | OK           | 695.000   | Seriell        |
| matrix_mul          | 2048       | OK           | 694.111   | Seriell        |
| matrix_mul_omp      | 512        | OK           | 1651.630  | OpenMP         |
| matrix_mul_omp      | 1024       | OK           | 1613.575  | OpenMP         |
| matrix_mul_omp      | 2000       | OK           | 1640.874  | OpenMP         |
| matrix_mul_omp      | 2048       | OK           | 1591.067  | OpenMP         |
| `matrix_mul_N`  | 512         | OK           | 6.359     | GPU (RTX 2070, OpenCL)     |
| `matrix_mul_N`  | 1024        | OK           | 47.761    | GPU (RTX 2070, OpenCL)     |
| `matrix_mul_N`  | 2000        | OK           | 381.980   | GPU (RTX 2070, OpenCL)     |
| `matrix_mul_N`  | 2048        | OK           | 290.469   | GPU (RTX 2070, OpenCL)     |


## Float
| Implementierung | Matrixgröße |Verifikation | Zeit [ms] | Bemerkung                  |
|-----------------|-------------|--------------|-----------|----------------------------|
|matrix_mul          | 512        | OK           | 282.317   | Seriell        |
|matrix_mul          | 1024       | OK           | 281.678   | Seriell        |
| matrix_mul          | 2000       | OK           | 279.857   | Seriell        |
| matrix_mul          | 2048       | OK           | 285.076   | Seriell        |
| matrix_mul_omp      | 512        | OK           | 721.991   | OpenMP         |
| matrix_mul_omp      | 1024       | OK           | 723.273   | OpenMP         |
| matrix_mul_omp      | 2000       | OK           | 762.201   | OpenMP         |
| matrix_mul_omp      | 2048       | OK           | 762.105   | OpenMP         |
| `matrix_mul_N`  | 512         | OK           | 5.983     | GPU (RTX 2070, OpenCL)     |
| `matrix_mul_N`  | 1024        | OK           | 45.594    | GPU (RTX 2070, OpenCL)     |
| `matrix_mul_N`  | 2000        | OK           | 222.784   | GPU (RTX 2070, OpenCL)     |
| `matrix_mul_N`  | 2048        | OK           | 263.740   | GPU (RTX 2070, OpenCL)     |

