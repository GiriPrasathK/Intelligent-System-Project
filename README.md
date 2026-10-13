# CPU vs Local GPU vs Cloud GPU: Conjugate Gradient

## About
This project evaluates the performance and scalability of the Conjugate Gradient (CG) method for solving large sparse linear systems across CPU, local GPU, and cloud GPU environments.

## Objectives
- Compare CG performance across CPU, local GPU, and cloud GPU.
- Analyze the impact of problem size and matrix sparsity on execution time.
- Measure speedup, convergence, and memory usage.
- Investigate data-transfer overhead and cloud execution cost.

## Methodology
- Generate sparse, symmetric positive-definite linear systems.
- Execute CG using compatible CPU and GPU implementations.
- Benchmark different problem sizes under consistent accuracy requirements.
- Analyze and visualize the experimental results.

## Technologies
Python, NumPy, SciPy, CUDA-compatible GPU libraries, and cloud GPU platforms (to be finalized).

## Project Status
**Current phase:** Literature review and experimental design.
