# Design

ParaForge has CPU kernels and two optional CUDA kernels (naive and shared-memory tiled). There is
no memory-pool code.

## Matrix

`paraforge::Matrix` is a dense row-major `float` matrix (`std::vector<float>` storage).
The constructor rejects shapes whose `rows * cols` would overflow before allocating.

## Kernel

`multiply_rows(a, b, c, row_begin, row_end)` (in `src/matrix.cpp`) computes rows
`[row_begin, row_end)` of `c = a * b` with a plain `i, j, k` loop and a `double` accumulator.
Both public kernels call it:

- `matmul_reference(a, b)`: one call over all rows, on the calling thread.
- `matmul_parallel(a, b, pool, grain)`: one call per row block, on pool workers.

The kernel is deliberately naive (no blocking, transposition or SIMD intrinsics). It is the
correctness baseline that later kernels are checked against, not a fast GEMM.

## ThreadPool

- Fixed number of `std::thread` workers, created in the constructor.
- One `std::queue<std::function<void()>>` guarded by one mutex; `submit` pushes and wakes one
  worker through a condition variable.
- `wait_all` blocks until the queue is empty and no task is running (`active_ == 0`), then
  rethrows the first exception any task threw since the previous `wait_all`. Workers catch task
  exceptions so one bad task cannot `std::terminate` the process.
- The destructor runs every queued task, then joins. If thread creation fails part-way, the
  constructor stops and joins the threads it already started before rethrowing.

`wait_all` waits for *all* work in the pool, so `matmul_parallel` assumes it has the pool to
itself while it runs.

## Row-block split

`matmul_parallel` cuts the `a.rows()` output rows into consecutive blocks of `grain` rows (the
last block may be shorter) and submits one task per block. Each task writes only its own rows of
`c` and only reads `a` and `b`, so tasks need no locking beyond the queue. If `submit` throws, the
function still waits for already-queued tasks before `c` goes out of scope, since they hold
references to it.

`grain` trades scheduling overhead against load balance:

- small grain: many tasks, more queue/mutex traffic, but idle workers can pick up leftovers;
- large grain: few tasks; once `ceil(rows / grain)` drops below the thread count some workers sit
  idle (e.g. 512 rows at grain 128 is only 4 tasks).

## Correctness

Every output element is computed by the same code, in the same `k` order, with the same `double`
accumulator, whichever thread runs it. So `matmul_parallel` is **bitwise** equal to
`matmul_reference`, and tests and the benchmark compare with exact equality (`memcmp`), not a
tolerance.

Tests (GoogleTest, `tests/`): matrix shape/layout/overflow, known products, identity, empty inner
dimension, shape-mismatch errors, pool behaviour (all tasks run, reuse, concurrency, exception
propagation, draining destructor), and parallel-vs-reference equality across shapes, grains and
pool sizes.

## Measurement

`paraforge_bench [size] [threads] [grain] [repeats]` builds two random `size x size` matrices
(fixed seed 42), starts the pool (not timed), runs both kernels once untimed and checks they
match, then times `repeats` runs of each kernel with `std::chrono::steady_clock` and prints the
median. Allocation of the output matrix is inside the timed region; freeing the previous result
is not. The timed results are checked against the warm-up result again before printing.

Speedup in `docs/results.md` is `cpu_single median / cpu_parallel median` from the same process
run, so each pair shares the same inputs, machine state and build.

## CUDA kernel (naive)

Built only with `-DPARAFORGE_ENABLE_CUDA=ON` (library `paraforge_cuda`, links `paraforge`).

- `src/matmul_naive_kernel.cuh`: the kernel. One thread computes one `C(row, col)` as a `float`
  dot product of row `row` of A and column `col` of B. Threads whose `(row, col)` falls outside
  `m x n` return at once, since the grid is rounded up to whole 16 x 16 blocks. Offsets are
  computed in `size_t`. The file has no host code so `bench/nvrtc_kernel_check.py` can JIT it.
- `src/matmul_cuda.cu`: `matmul_cuda_naive(a, b, timings)` (and `matmul_cuda_tiled`, below,
  which shares this host code through `run_matmul`). Every call allocates three device
  buffers, copies A and B in, launches `ceil(n/16) x ceil(m/16)` blocks, copies C back and frees.
  Empty shapes (`m`, `n` or `k` = 0) return the zero/empty result without touching the GPU.
- Errors: every CUDA call is checked and turned into `std::runtime_error` with
  `cudaGetErrorString`; launch-config errors are caught by `cudaGetLastError` right after the launch,
  faults during the kernel by the following `cudaMemcpy`. Device buffers and events are RAII, so
  a throw does not leak them. Shapes the `int` kernel arguments or the 65535-block `grid.y` limit
  cannot cover (any dimension above `INT_MAX - 16`, or `m` above `65535 * 16`) throw
  `std::invalid_argument` before any allocation.
- Timing: four CUDA events on the default stream split the call into H2D copy, kernel and D2H
  copy (`CudaTimings`). Device allocation/free are outside all three; the benchmark's
  `end_to_end_ms` wall clock includes them. CUDA context creation is excluded by an untimed
  warm-up call.

## CUDA kernel (tiled)

`src/matmul_tiled_kernel.cuh`, called by `matmul_cuda_tiled(a, b, timings)`. Same host code,
launch shape (`ceil(n/16) x ceil(m/16)` blocks of 16 x 16 threads), errors and timing as the naive
kernel: both share one `run_matmul` in `src/matmul_cuda.cu`, which branches to the named launch.

- Each block owns one 16 x 16 tile of C. It walks `k` in steps of 16. Per step every thread loads
  one element of A (`row`, `t + tx`) and one of B (`t + ty`, `col`) into two `__shared__` 16 x 16
  arrays, `__syncthreads()`, then adds 16 products from shared memory, then `__syncthreads()` again
  so no thread overwrites a tile another thread is still reading.
- Partial tiles: a load whose row or column is out of range (past `m`, `n` or `k`) writes `0.0f`
  instead of reading global memory. Padded A and B entries are both 0 past `k`, so they add exact
  zeros. Out-of-range threads do **not** return early: they must still load their share of the tile
  and reach both barriers (a thread skipping `__syncthreads()` is undefined behaviour and in practice
  hangs or corrupts the block). Only threads with `row < m && col < n` store.
- `static_assert(kBlock == kTile)` ties the launch shape to the tile size.
- Each output is summed in the same `k` order as the naive kernel; every local check so far gave
  identical max errors for both kernels, but bitwise equality is not guaranteed (FMA contraction is
  the compiler's choice).
- Why only ~1.3x (see [results.md](results.md#gpu)): likely because the naive kernel already gets
  much of its A/B reuse from L1/L2 on this GPU, and the tiled inner loop does two shared-memory loads
  per multiply-add, so it is bound by shared-memory bandwidth. Not profiled. Larger gains need each
  thread to compute several outputs from registers (register blocking), wider loads, or cuBLAS.

How the local NVRTC check validates it: `bench/nvrtc_kernel_check.py` gives A, B and C a NaN tail
on the device. Removing either load guard makes the last row read the NaN tail (checks fail with
`max_abs_err=nan`), removing the store guard trips the "wrote past C" check, and dropping one
product per tile fails the tolerance (all tried by editing the kernel by hand).

### Correctness tolerance

The GPU sums in `float` (and may fuse multiply-adds), the reference in `double`, so results are
not bitwise equal. Tests and the benchmark accept `|gpu - cpu| <= 1e-5 * k` per element, for
inputs in `[-1, 1]`. The bound applies to both kernels. Measured errors on an RTX 3050 are over 100x below it (e.g. 1024^3:
max error ~8e-5 vs tolerance ~1e-2), while dropping one product term from the sum or removing
the bounds check makes the check fail (tried by editing the kernel by hand).
