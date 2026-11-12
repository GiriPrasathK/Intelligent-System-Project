# Week 9 – Cloud-Based GPU Computing

## 1. Objective

Deploy the GPU-based Conjugate Gradient project on a cloud computing platform.

## 2. Cloud Environment

The project was deployed using Google Colab Free with the following configuration:

- GPU: NVIDIA Tesla T4
- GPU memory: 14.56 GB
- Programming language: Python
- GPU programming: Numba CUDA
- Numerical precision: float64
- Sparse matrix format: CSR
- Stopping tolerance: 1e-6

## 3. Implementation

The existing GPU-based Conjugate Gradient solver was reused and executed in the cloud environment.

The test problem was a sparse, symmetric positive-definite tridiagonal matrix:

A has 2 on its main diagonal and -1 on its adjacent diagonals.

The known solution was a vector of ones, and the right-hand side was generated using b = Ax.

The solver used CUDA kernels for sparse matrix-vector multiplication, dot products and vector updates. The CPU controlled the iterative solver loop, including scalar calculations and synchronization.

An interactive Gradio application was developed to execute the solver for different matrix sizes and display benchmark results.

## 4. Experimental Results

The solver was evaluated using five matrix sizes.

| Matrix size | NNZ | Iterations | Setup and transfer (ms) | Solver time (ms) | True relative residual |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 2,998 | 500 | 1.445 | 377.818 | 3.6693e-13 |
| 5,000 | 14,998 | 2,500 | 1.238 | 1,843.105 | 7.1663e-12 |
| 10,000 | 29,998 | 5,000 | 1.361 | 3,125.859 | 1.9503e-11 |
| 20,000 | 59,998 | 10,000 | 1.450 | 5,704.081 | 4.2566e-11 |
| 50,000 | 149,998 | 25,000 | 2.221 | 17,341.251 | 1.6791e-10 |

All five experiments converged successfully with true relative residuals below the specified tolerance of 1e-6.

## 5. Performance Analysis

The solver execution time increased with matrix size and iteration count. The largest tested problem, containing 50,000 unknowns and 149,998 non-zero elements, completed in 17,341.251 ms.

The measured setup and transfer time remained between approximately 1.2 and 2.2 ms.

The solver time includes iterative execution and CPU–GPU synchronization overhead. It should not be interpreted as pure CUDA kernel execution time.

These results represent initial measurements from one run per matrix size. Repeated experiments are required for more reliable timing comparisons.

## 6. Conclusion

The Conjugate Gradient implementation was successfully deployed and executed on the Google Colab Tesla T4 GPU. The solver produced accurate solutions and converged for all five tested matrix sizes, up to 50,000 unknowns.

The cloud deployment and initial performance evaluation demonstrate that the selected cloud environment supports the project's GPU computing requirements.

**Week 9 Status: Completed – GPU-based project successfully deployed and benchmarked on Google Colab.**
