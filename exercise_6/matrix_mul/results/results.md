## \# Matrix Multiplication -- Optimization Results

## 1. Kernel Design (`matrix_mul.cl`)

### 1.1 Multiple Columns per Work-Item

**Concept:**\
Each work-item computes **`COLS_PER_THREAD`** columns of the output
matrix **C**.

**Indexing logic:**

``` c
int row       = get_global_id(0);
int col_group = get_global_id(1);
int col0      = col_group * COLS_PER_THREAD;
```

------------------------------------------------------------------------

## 2. Kernel Parameters & Optimizations

### 2.1 Configurable `COLS_PER_THREAD`

Within the kernel:

``` c
#ifndef COLS_PER_THREAD
#define COLS_PER_THREAD 2
#endif

#if COLS_PER_THREAD < 1 || COLS_PER_THREAD > 4
#error "COLS_PER_THREAD must be between 1 and 4"
#endif
```

The actual value is provided from the host using:

    -DCOLS_PER_THREAD=<value>

------------------------------------------------------------------------

### 2.2 Loop Unrolling

Inner multiplication loop:

``` c
#pragma unroll 4
for (int k = 0; k < M; ++k) {
    ...
}
```

**Purpose:** - Increase instruction-level parallelism (ILP)\
- Reduce memory-latency bottlenecks

------------------------------------------------------------------------

## 3. Build Options

### Float Kernel:

    -DCOLS_PER_THREAD=<value> -DTILE_X=<value> -DTILE_Y=<value>
    -cl-mad-enable -cl-fast-relaxed-math

### Double Kernel:

    -DUSE_DOUBLE=1
    -DCOLS_PER_THREAD=<value> -DTILE_X=<value> -DTILE_Y=<value>
    -cl-mad-enable -cl-fast-relaxed-math

### Effects:

-   `-cl-mad-enable` --- enables fused multiply-add (FMA)\
-   `-cl-fast-relaxed-math` --- faster, less strict floating-point
    behavior\
-   `-D...` --- passes tuning constants to the kernel

------------------------------------------------------------------------

## 4. NDRange & Work-Group Configuration (Host)

### 4.1 Parameter Sweep

The host autotuner iterates through the following candidate sets:

-   `COLS_PER_THREAD ∈ {1, 2, 4}`
-   `TILE_X ∈ {4, 8, 16, 32}`
-   `TILE_Y ∈ {1, 2, 4, 8, 16, 32}`

------------------------------------------------------------------------

### 4.2 Work-Group Size Constraint

``` c
size_t wg_size = (size_t)tile_x * (size_t)tile_y;
if (wg_size > max_wg_size) continue;
```

------------------------------------------------------------------------

### 4.3 Global & Local Work Sizes

**Number of column groups:**

``` c
const size_t num_col_groups =
    ((size_t)K + cols_per_thread - 1) / cols_per_thread;
```

**Global size, rounded up to tile dimensions:**

``` c
global_work_size[0] =
    ((size_t)N + tile_x - 1) / tile_x * tile_x;

global_work_size[1] =
    (num_col_groups + tile_y - 1) / tile_y * tile_y;
```

**Local size:**

``` c
const size_t local_work_size[2] = { tile_x, tile_y };
```

------------------------------------------------------------------------

## 5. Auto-Tuning Procedure

For every parameter combination:

1.  Build the program with the corresponding compile-time defines\
2.  Create the kernel (float or double version)\
3.  Set arguments: `A, B, C, N, M, K`\
4.  Execute the kernel multiple times (e.g., 3 runs)\
5.  Measure execution time using `omp_get_wtime()`\
6.  Compute the average execution time\
7.  Track the fastest configuration

Finally:

-   Output the best parameter combination\
-   Optionally rebuild using the best parameters and verify correctness
    (e.g., checking `C[0,0]`)

------------------------------------------------------------------------

## 6. Best Parameters (for the tested AMD GPU)

The autotuning process determined the best configuration as:

  Parameter           Value
  ------------------- ---------
  `COLS_PER_THREAD`   **2**
  `TILE_X`            **8**
  `TILE_Y`            **32**
  `WG_SIZE`           **256**

These values are optimal for the specific AMD GPU used in testing and
may be hard-coded for deployment.

