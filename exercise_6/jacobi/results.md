# Jacobi

## Work Group Size
Same as for the RTX2070, the workgroupsize of {1, 256} was the fastest on the AMD cluster. Choosing not optimal workgroupsizes impacts the AMD system significantly more. The speedup factor going from {8, 32} to {1, 256} is about 1.35 for NVIDIA, whereas it is 5.01 for AMD (float, IT=1000, N=4096).

## Problem Size Scaling

Here we are using the results from the NVIDIA cluster as a baseline and compare the results of the AMD cluster with it (a Ratio <1 means AMD is faster, >1 means AMD is slower). Workgroupsize was {1, 256}, the ideal configuration for both clusters.

### N=2048

| IT | NVIDIA Double | AMD Double | Ratio (Double) | NVIDIA Float | AMD Float | Ratio (Float) |
|------|-------------|------------|----------------|--------------|-----------|---------------|
| 10   | 26 ms       | 16 ms      | 1.63×          | 14 ms        | 14 ms     | 1.00×         |
| 100  | 51 ms       | 95 ms      | 1.87×          | 30 ms        | 91 ms     | 3.03×         |
| 1000 | 283 ms      | 876 ms     | 3.10×          | 164 ms       | 860 ms    | 5.24×         |

### N=4096

| IT | NVIDIA Double | AMD Double | Ratio (Double) | NVIDIA Float | AMD Float | Ratio (Float) |
|------|-------------|------------|----------------|--------------|-----------|---------------|
| 10   | 100 ms      | 54 ms      | 0.54×          | 52 ms        | 47 ms     | 0.90×         |
| 100  | 193 ms      | 380 ms     | 1.97×          | 105 ms       | 358 ms    | 3.41×         |
| 1000 | 1,127 ms    | 3,625 ms   | 3.22×          | 655 ms       | 3,498 ms  | 5.34×         |

We can see that for the larger matrix and few iterations, AMD is faster. The scaling on the AMD system is significantly worse when the Iterations get higher. This suggests that the kernel launch overhead is higher for AMD.

Another interersting difference between the two clusters is the impact of using float and double: on AMD the execution time stays the same, regardless of precision. On NVIDIA, double is significantly slower: up to x1.8 the time of float.