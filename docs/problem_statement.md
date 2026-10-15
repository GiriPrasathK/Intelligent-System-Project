
## Title
Performance and Scalability Analysis of the Conjugate Gradient Method Across CPU, Local GPU, and Cloud GPU Environments

## 1. Problem Statement
The Conjugate Gradient (CG) method is an iterative numerical method used to solve large sparse symmetric positive-definite linear systems. Its computational workload includes operations such as sparse matrix-vector multiplication, vector updates, and inner products.

This project investigates the performance of the CG method across CPU, local GPU, and cloud GPU environments. It aims to understand how execution time, workload size, memory usage, and computational overhead affect performance.

## 2. Objectives
- Compare CG execution time across CPU, local GPU, and cloud GPU environments.
- Study how problem size affects execution time and GPU acceleration.
- Evaluate convergence and residual error to ensure numerical correctness.
- Investigate CPU–GPU data-transfer overhead.
- Examine cloud execution cost and scalability where feasible.

## 3. Research Questions
1. How does CG execution time differ across CPU, local GPU, and cloud GPU environments?
2. How does increasing matrix size affect execution time and speedup?
3. What impact does CPU–GPU data transfer have on total execution time?
4. Does cloud GPU execution provide a useful performance benefit after considering overhead and cost?

## 4. Proposed Methodology
1. Generate sparse symmetric positive-definite test matrices.
2. Implement or select compatible CPU and GPU CG solvers.
3. Validate solutions using a consistent convergence tolerance.
4. Benchmark multiple problem sizes and repeat measurements.
5. Compare execution time, speedup, convergence, and available resource measurements.
6. Present the results using tables and plots.

## 5. Expected Outcome
The project will provide an experimental comparison of CG performance across the selected environments. The results will identify workload sizes and execution conditions under which each environment is effective.

## 6. Current Status
Week 1: Studying GPU computing fundamentals and defining the project problem. Hardware and software configurations are yet to be finalized.
