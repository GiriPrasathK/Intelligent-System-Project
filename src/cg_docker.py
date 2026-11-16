import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from numba import cuda


# --------------------------------------------------
# 1. CUDA Kernels
# --------------------------------------------------

@cuda.jit
def csr_spmv(values, col_indices, row_ptr, x, y, n):
    row = cuda.grid(1)

    if row < n:
        row_sum = 0.0
        for j in range(row_ptr[row], row_ptr[row + 1]):
            row_sum += values[j] * x[col_indices[j]]
        y[row] = row_sum


@cuda.jit
def dot_product_kernel(a, b, partial, n):
    tid = cuda.threadIdx.x
    idx = cuda.grid(1)

    shared = cuda.shared.array(256, dtype=np.float64)

    value = 0.0
    if idx < n:
        value = a[idx] * b[idx]

    shared[tid] = value
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2

    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial[cuda.blockIdx.x] = shared[0]


@cuda.jit
def vector_update(a, b, result, alpha, operation, n):
    i = cuda.grid(1)

    if i < n:
        if operation == 0:
            result[i] = a[i] + alpha * b[i]
        elif operation == 1:
            result[i] = a[i] - alpha * b[i]
        elif operation == 2:
            result[i] = alpha * a[i]


# --------------------------------------------------
# 2. GPU Conjugate Gradient
# --------------------------------------------------

def gpu_conjugate_gradient(
    values, col_indices, row_ptr, b,
    tol=1e-6, max_iter=10000
):
    n = len(b)
    dtype = np.float64

    d_b = cuda.to_device(b.astype(dtype))
    d_x = cuda.to_device(np.zeros(n, dtype=dtype))

    d_r = cuda.device_array(n, dtype=dtype)
    d_p = cuda.device_array(n, dtype=dtype)
    d_Ap = cuda.device_array(n, dtype=dtype)

    threads = 256
    blocks = (n + threads - 1) // threads

    d_partial = cuda.device_array(blocks, dtype=dtype)

    d_r.copy_to_device(d_b)
    d_p.copy_to_device(d_b)

    dot_product_kernel[blocks, threads](
        d_r, d_r, d_partial, n
    )
    rs_old = np.sum(d_partial.copy_to_host(), dtype=np.float64)

    initial_norm = np.sqrt(rs_old)

    if initial_norm == 0:
        return np.zeros(n), 0, 0.0

    for iteration in range(max_iter):

        csr_spmv[blocks, threads](
            values, col_indices, row_ptr, d_p, d_Ap, n
        )

        dot_product_kernel[blocks, threads](
            d_p, d_Ap, d_partial, n
        )
        pAp = np.sum(d_partial.copy_to_host(), dtype=np.float64)

        if pAp <= 0:
            raise ValueError("Matrix may not be positive definite.")

        alpha = rs_old / pAp

        vector_update[blocks, threads](
            d_x, d_p, d_x, alpha, 0, n
        )

        vector_update[blocks, threads](
            d_r, d_Ap, d_r, alpha, 1, n
        )

        dot_product_kernel[blocks, threads](
            d_r, d_r, d_partial, n
        )
        rs_new = np.sum(d_partial.copy_to_host(), dtype=np.float64)

        residual_norm = np.sqrt(rs_new)

        if residual_norm <= tol * initial_norm:

            csr_spmv[blocks, threads](
                values, col_indices, row_ptr, d_x, d_Ap, n
            )

            true_r = b - d_Ap.copy_to_host()
            true_norm = np.linalg.norm(true_r)

            if true_norm <= tol * initial_norm:
                return d_x.copy_to_host(), iteration + 1, true_norm

            d_r.copy_to_device(true_r)
            d_p.copy_to_device(true_r)

            dot_product_kernel[blocks, threads](
                d_r, d_r, d_partial, n
            )
            rs_old = np.sum(d_partial.copy_to_host(), dtype=np.float64)
            continue

        beta = rs_new / rs_old

        vector_update[blocks, threads](
            d_r, d_p, d_p, beta, 0, n
        )

        rs_old = rs_new

    return d_x.copy_to_host(), max_iter, np.sqrt(rs_old)


# --------------------------------------------------
# 3. Generate Sparse SPD Matrix
# --------------------------------------------------

N = 1000

main_diag = 2.0 * np.ones(N)
off_diag = -1.0 * np.ones(N - 1)

A = diags(
    [off_diag, main_diag, off_diag],
    offsets=[-1, 0, 1],
    format="csr"
)

x_true = np.ones(N, dtype=np.float64)
b = np.asarray(A @ x_true, dtype=np.float64)


# --------------------------------------------------
# 4. CPU CG
# --------------------------------------------------

print("\nCPU Conjugate Gradient")
print("----------------------")

start = time.perf_counter()

x_cpu, cpu_info = cg(
    A, b,
    x0=np.zeros(N),
    rtol=1e-6,
    atol=0.0,
    maxiter=10000
)

cpu_time = time.perf_counter() - start

cpu_residual = np.linalg.norm(b - A @ x_cpu)

print("Convergence status:", cpu_info)
print("Execution time:", cpu_time, "seconds")
print("True residual:", cpu_residual)
print("Validation:", np.allclose(x_cpu, x_true, atol=1e-4))


# --------------------------------------------------
# 5. GPU CG
# --------------------------------------------------

print("\nGPU Conjugate Gradient")
print("----------------------")

values = cuda.to_device(A.data.astype(np.float64))
col_indices = cuda.to_device(A.indices.astype(np.int32))
row_ptr = cuda.to_device(A.indptr.astype(np.int32))

start = time.perf_counter()

x_gpu, iterations, gpu_residual = gpu_conjugate_gradient(
    values, col_indices, row_ptr, b,
    tol=1e-6,
    max_iter=10000
)

cuda.synchronize()
gpu_time = time.perf_counter() - start

true_gpu_residual = np.linalg.norm(b - A @ x_gpu)

print("Iterations:", iterations)
print("Execution time:", gpu_time, "seconds")
print("Returned residual:", gpu_residual)
print("True residual:", true_gpu_residual)
print("Validation:", np.allclose(x_gpu, x_true, atol=1e-4))


# --------------------------------------------------
# 6. CPU vs GPU Comparison
# --------------------------------------------------

print("\nCPU vs GPU Comparison")
print("---------------------")
print("CPU solution error:", np.linalg.norm(x_cpu - x_true))
print("GPU solution error:", np.linalg.norm(x_gpu - x_true))
print("CPU-GPU difference:", np.linalg.norm(x_cpu - x_gpu))

if gpu_time > 0:
    print("GPU speedup:", cpu_time / gpu_time, "x")
