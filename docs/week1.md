# Week 1: Introduction to GPU Computing

## 1. Introduction
GPU computing uses Graphics Processing Units (GPUs) to execute computational tasks. Although GPUs were originally developed for graphics processing, they are also widely used in scientific computing, numerical simulations, machine learning, and other computationally intensive applications.

General-Purpose computing on Graphics Processing Units (GPGPU) refers to using GPUs for computation beyond traditional graphics rendering.

## 2. CPU vs GPU
A Central Processing Unit (CPU) is designed for versatile processing, complex control logic, and low-latency execution. A GPU is designed to provide high computational throughput by executing many operations in parallel.

| Feature | CPU | GPU |
|---|---|---|
| Primary emphasis | Low latency and versatility | High throughput |
| Processing resources | Relatively few powerful cores | Many parallel processing resources |
| Suitable workloads | Sequential tasks and complex control | Large parallel workloads |
| Important consideration | CPU parallelization | Parallel execution and data-transfer overhead |

Neither processor is universally faster. Performance depends on workload size, implementation, hardware, and overhead.

## 3. Parallel Computing
Parallel computing involves executing multiple parts of a computation concurrently.

Data parallelism occurs when the same operation is applied to multiple data elements. For example, two arrays can be added element by element because each output element can be calculated independently.

Parallel execution can improve performance, but dependencies, synchronization, and overhead can limit the speedup.

## 4. Relevance to the Conjugate Gradient Project
The Conjugate Gradient method solves systems of linear equations of the form:

Ax = b

For standard CG, A is a symmetric positive-definite matrix. Important computational operations include sparse matrix-vector multiplication, vector updates, and inner products.

Sparse matrix-vector multiplication is a candidate for parallel execution because different output rows can often be calculated independently.

## 5. Project Application
The project will compare CG execution across CPU, local GPU, and cloud GPU environments. It will investigate execution time, workload size, numerical accuracy, and computational overhead.

## 6. Key Takeaways
- GPUs are useful for workloads with sufficient parallelism.
- CPUs are well suited to versatile processing and complex control.
- GPU acceleration does not guarantee a speedup for small workloads.
- Sparse matrix-vector multiplication is an important operation in CG.
- Fair benchmarking requires consistent inputs and numerical accuracy requirements.

## 7. Status
Completed the initial study of GPU computing, CPU–GPU differences, and parallel computing concepts. Practical implementation and benchmarking remain future tasks.
