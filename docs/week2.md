# Week 2: GPU Architecture and Parallel Programming

## 1. Objective

To study GPU architecture, Streaming Multiprocessors (SMs), CUDA thread organization, SIMT execution, parallelism, and synchronization.

## 2. GPU Architecture and Streaming Multiprocessors

A GPU contains multiple Streaming Multiprocessors (SMs). These units execute groups of GPU threads and use hardware resources to schedule and execute parallel workloads.

The number of SMs and their internal resources depend on the GPU model and architecture.

## 3. Threads, Blocks, and Grids

CUDA organizes kernel execution using three levels:

- **Thread:** An individual execution instance.
- **Block:** A group of threads that can cooperate and synchronize.
- **Grid:** The collection of blocks launched for a kernel.

The hierarchy is:

Grid → Blocks → Threads

For a one-dimensional grid, the global thread index is commonly calculated as:

`i = blockIdx.x * blockDim.x + threadIdx.x`

A bounds check is needed when the calculated index may exceed the input size.

## 4. SIMT Execution Model

SIMT stands for Single Instruction, Multiple Threads. Threads execute similar instructions on different data elements and are scheduled in groups called warps.

Branch divergence occurs when threads in the same warp take different execution paths, potentially reducing execution efficiency.

## 5. Parallelism and Synchronization

Parallelism allows independent calculations to execute concurrently. However, some operations depend on results produced by other threads.

CUDA provides `__syncthreads()` to synchronize threads within the same thread block. It is not a general grid-wide synchronization mechanism.

## 6. Connection to the Conjugate Gradient Project

The Conjugate Gradient method solves systems of linear equations:

`Ax = b`

For standard CG, the matrix must be symmetric positive definite.

Important operations include:

- Sparse matrix-vector multiplication
- Vector updates
- Dot products and reductions

Different matrix rows or vector elements can often be processed in parallel. Dot products require reduction operations, and dependencies between CG iterations limit how much of the algorithm can execute concurrently.

## 7. Learning Outcomes

- Understood the role of SMs in GPU execution.
- Learned the grid, block, and thread hierarchy.
- Understood global thread indexing and bounds checking.
- Studied SIMT, warps, and branch divergence.
- Identified synchronization and reduction challenges in CG.

## 8. Status

Week 2 theory completed. Practical CUDA implementation and hardware-specific experiments are pending.
