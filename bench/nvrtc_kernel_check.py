"""Runs src/matmul_naive_kernel.cuh and src/matmul_tiled_kernel.cuh on the local GPU without
nvcc or a host C++ compiler.

For machines with an NVIDIA driver but no CUDA toolkit (e.g. Windows without MSVC, which nvcc
needs). It JIT-compiles the exact kernel sources with NVRTC, launches them through the driver API,
checks them against a float64 NumPy reference and times each kernel with CUDA events, both on the
same inputs. It does not build or test src/matmul_cuda.cu (the C++ host side); use the CMake CUDA
build for that.

Needs: NVIDIA driver, NumPy, and NVRTC from a pip wheel matching the driver's CUDA version, e.g.
    python -m venv .venv && .venv/Scripts/pip install numpy "nvidia-cuda-nvrtc-cu12==12.2.*"
Usage: python bench/nvrtc_kernel_check.py [m k n repeats] [--csv FILE]
    timed shape defaults to 1024 1024 1024 5; --csv appends one row per kernel to FILE.
"""

import argparse
import ctypes
import datetime
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = 16  # must match kBlock in src/matmul_cuda.cu
KERNELS = [("naive", "matmul_naive_kernel.cuh", b"matmul_naive_kernel"),
           ("tiled", "matmul_tiled_kernel.cuh", b"matmul_tiled_kernel")]
CSV_HEADER = ("timestamp_utc,device,driver_cuda,nvrtc,kernel,m,k,n,repeats,kernel_ms_median,"
              "kernel_ms_min,kernel_ms_max,kernel_gflops,max_abs_err\n")


def load_nvrtc():
    import nvidia.cuda_nvrtc  # namespace package from the pip wheel

    base = list(nvidia.cuda_nvrtc.__path__)[0]
    pattern = "bin/nvrtc*.dll" if os.name == "nt" else "lib/libnvrtc*.so.*"
    found = glob.glob(os.path.join(base, pattern))
    libs = [p for p in found if "builtins" not in p]
    if not libs:
        sys.exit(f"error: no NVRTC library under {base}")
    for p in found:
        if "builtins" in p:
            ctypes.CDLL(p)  # NVRTC loads this by bare name later; preloading makes that find it
    return ctypes.CDLL(libs[0])


cuda = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
nvrtc = load_nvrtc()


def check_cu(err, what):
    if err != 0:
        msg = ctypes.c_char_p()
        cuda.cuGetErrorString(err, ctypes.byref(msg))
        sys.exit(f"error: {what}: CUDA driver error {err} ({msg.value.decode()})")


def compile_cubin(filename, arch):
    with open(os.path.join(ROOT, "src", filename), "rb") as f:
        src = f.read()
    prog = ctypes.c_void_p()
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, filename.encode(), 0, None, None) == 0
    opts = (ctypes.c_char_p * 1)(f"--gpu-architecture=sm_{arch}".encode())
    rc = nvrtc.nvrtcCompileProgram(prog, 1, opts)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        sys.exit(f"error: NVRTC compile of {filename} failed ({rc}):\n{log.value.decode()}")
    if log.value.strip():
        print(f"nvrtc log ({filename}): {log.value.decode().strip()}")
    nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size))
    cubin = ctypes.create_string_buffer(size.value)
    assert nvrtc.nvrtcGetCUBIN(prog, cubin) == 0
    return cubin


def alloc(nbytes):
    ptr = ctypes.c_uint64()
    check_cu(cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 1))), "cuMemAlloc")
    return ptr


def upload(host, what):
    ptr = alloc(host.nbytes)
    check_cu(cuda.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(host.ctypes.data), ctypes.c_size_t(host.nbytes)), what)
    return ptr


def run(func, a, b, guard=64):
    """Launches the kernel once; returns (C, kernel_ms). Fails if it writes past C.

    A and B get a NaN tail too: a kernel that uses a value read past the end of either poisons C
    with NaN, which the tolerance check rejects (reads past a row inside A/B give wrong values).
    """
    m, k = a.shape
    n = b.shape[1]
    nan_tail = np.full(guard, np.nan, dtype=np.float32)
    c_host = np.full(m * n + guard, np.nan, dtype=np.float32)  # NaN tail catches out-of-bounds writes
    d_a = upload(np.concatenate([a.ravel(), nan_tail]), "HtoD A")
    d_b = upload(np.concatenate([b.ravel(), nan_tail]), "HtoD B")
    d_c = upload(c_host, "HtoD C")
    args = [d_a, d_b, d_c, ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k)]
    params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(x) for x in args])
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    check_cu(cuda.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
    check_cu(cuda.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate")
    check_cu(cuda.cuEventRecord(start, None), "cuEventRecord")
    check_cu(cuda.cuLaunchKernel(func, (n + BLOCK - 1) // BLOCK, (m + BLOCK - 1) // BLOCK, 1,
                                 BLOCK, BLOCK, 1, 0, None, params, None), "cuLaunchKernel")
    check_cu(cuda.cuEventRecord(stop, None), "cuEventRecord")
    check_cu(cuda.cuEventSynchronize(stop), "kernel")
    ms = ctypes.c_float()
    check_cu(cuda.cuEventElapsedTime(ctypes.byref(ms), start, stop), "cuEventElapsedTime")
    check_cu(cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(c_host.ctypes.data), d_c, ctypes.c_size_t(c_host.nbytes)), "DtoH C")
    for p in (d_a, d_b, d_c):
        cuda.cuMemFree_v2(p)
    cuda.cuEventDestroy_v2(start)
    cuda.cuEventDestroy_v2(stop)
    if not np.isnan(c_host[m * n:]).all():
        sys.exit(f"error: kernel wrote past C for shape {m}x{k}x{n}")
    return c_host[: m * n].reshape(m, n), ms.value


def check_shape(funcs, m, k, n, rng):
    """Runs every kernel on the same random inputs; returns {name: (ok, kernel_ms, max_abs_err)}."""
    a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
    want = a.astype(np.float64) @ b.astype(np.float64)
    tol = 1e-5 * k  # same bound as paraforge_cuda_bench and the GoogleTest suite
    results = {}
    for name, func in funcs.items():
        got, ms = run(func, a, b)
        err = float(np.max(np.abs(got.astype(np.float64) - want)))
        status = "ok" if err <= tol else "FAIL"  # NaN compares False, so it fails too
        print(f"check {name} {m}x{k}x{n}: max_abs_err={err:.3g} tol={tol:.3g} {status}")
        results[name] = (status == "ok", ms, err)
    return results


def main():
    parser = argparse.ArgumentParser(description="NVRTC check and kernel timing of both kernels")
    parser.add_argument("timed", nargs="*", type=int, metavar="m k n repeats")
    parser.add_argument("--csv", help="append one row per kernel to this CSV file")
    args = parser.parse_args()
    timed = args.timed or [1024, 1024, 1024, 5]
    if len(timed) != 4 or min(timed) <= 0:
        sys.exit("usage: nvrtc_kernel_check.py [m k n repeats] [--csv FILE]")
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    check_cu(cuda.cuInit(0), "cuInit")
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    check_cu(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    major, minor, ver = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)  # COMPUTE_CAPABILITY_MAJOR
    cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)  # COMPUTE_CAPABILITY_MINOR
    name = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name, 256, dev)
    cuda.cuDriverGetVersion(ctypes.byref(ver))
    nv_major, nv_minor = ctypes.c_int(), ctypes.c_int()
    nvrtc.nvrtcVersion(ctypes.byref(nv_major), ctypes.byref(nv_minor))
    device, driver_cuda = name.value.decode(), f"{ver.value // 1000}.{ver.value % 1000 // 10}"
    nvrtc_ver = f"{nv_major.value}.{nv_minor.value}"
    print(f"{stamp} device: {device} sm_{major.value}{minor.value}, driver CUDA {driver_cuda}, "
          f"NVRTC {nvrtc_ver}")
    check_cu(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
    check_cu(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    funcs = {}
    for kname, filename, symbol in KERNELS:
        module, func = ctypes.c_void_p(), ctypes.c_void_p()
        cubin = compile_cubin(filename, f"{major.value}{minor.value}")
        check_cu(cuda.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
        check_cu(cuda.cuModuleGetFunction(ctypes.byref(func), module, symbol), "cuModuleGetFunction")
        funcs[kname] = func

    rng = np.random.default_rng(42)
    # Tiny, odd, partial tiles (non-multiples of 16 in m, k and n), exact tiles, 1 x k and k x 1.
    shapes = [(1, 1, 1), (2, 3, 2), (15, 15, 15), (16, 16, 16), (17, 17, 17), (32, 48, 16),
              (64, 64, 64), (17, 33, 15), (37, 19, 53), (31, 1, 33), (1, 300, 70), (70, 300, 1),
              (130, 257, 65), (256, 512, 128), (1000, 700, 300)]
    ok = True
    for m, k, n in shapes:
        ok = all(good for good, _, _ in check_shape(funcs, m, k, n, rng).values()) and ok

    m, k, n, repeats = timed
    times = {kname: [] for kname in funcs}
    worst = {kname: 0.0 for kname in funcs}
    for r in range(repeats + 1):  # round 0 is an untimed warm-up; kernels alternate every round
        for kname, (good, ms, err) in check_shape(funcs, m, k, n, rng).items():
            ok = ok and good
            worst[kname] = max(worst[kname], err)
            if r:
                times[kname].append(ms)
    rows = []
    for kname, t in times.items():
        kernel_ms = float(np.median(t))
        gflops = 2.0 * m * n * k / (kernel_ms * 1e6)
        print(f"timed {kname} {m}x{k}x{n}: kernel_ms median={kernel_ms:.3f} (min {min(t):.3f}, "
              f"max {max(t):.3f}) over {repeats} runs, {gflops:.1f} GFLOPS")
        rows.append(f"{stamp},{device},{driver_cuda},{nvrtc_ver},{kname},{m},{k},{n},{repeats},"
                    f"{kernel_ms:.3f},{min(t):.3f},{max(t):.3f},{gflops:.1f},{worst[kname]:.3g}\n")
    print(f"tiled speedup over naive (kernel only, medians): "
          f"{np.median(times['naive']) / np.median(times['tiled']):.2f}x")
    print("PASS" if ok else "FAIL")
    if args.csv and ok:  # never record numbers from a run that failed a check
        new_file = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            f.write((CSV_HEADER if new_file else "") + "".join(rows))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
