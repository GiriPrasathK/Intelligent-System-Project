# Week 3: Introduction to CUDA Programming

## 1. Objective

Understand the CUDA programming model, CUDA kernels, thread indexing, host and device execution, and basic CUDA programming.

## 2. Concepts Learned

- CUDA is NVIDIA's parallel computing platform and programming model.
- The host refers to the CPU side of a CUDA application.
- The device refers to the GPU.
- A CUDA kernel executes on the GPU using multiple threads.
- Thread indexing assigns work to individual GPU threads.
- Host-to-device and device-to-host memory transfers move data between CPU and GPU memory.

## 3. Practical Exercise: Vector Addition

**Program:** CUDA Vector Addition

The objective of this practical is to perform element-wise addition of two vectors using GPU parallelism.

The operation is:

C[i] = A[i] + B[i]

The program performs the following operations:

1. Initializes two input vectors on the CPU.
2. Allocates memory on the GPU.
3. Copies input vectors from host memory to device memory.
4. Launches a CUDA kernel to perform vector addition.
5. Synchronizes GPU execution.
6. Copies the result back to CPU memory.
7. Validates the computed output.

## 4. CUDA Kernel and Thread Indexing

The CUDA kernel uses the following indexing expression:

i = blockIdx.x * blockDim.x + threadIdx.x

Each thread calculates one element of the output vector.

A bounds check (`i < n`) ensures that threads with indices outside the vector size do not access invalid memory locations.

## 5. Thread Configuration

- Number of input elements: 1,000
- Threads per block: 256
- Number of blocks: 4
- Total thread positions launched: 1,024
- Valid elements processed: 1,000

The number of blocks is calculated as:

Blocks = ceil(1000 / 256) = 4

The additional 24 thread positions are handled using the bounds check.

## 6. Experimental Environment

- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- CUDA Toolkit: 13.0
- CUDA Compiler: NVCC
- Programming Language: CUDA C++

## 7. Execution Results

The program was successfully compiled and executed on the NVIDIA Tesla T4 GPU.

**Output:**

```text
Vector addition: PASS
Elements: 1000
Threads per block: 256
Blocks: 4
C[0] = 0
C[10] = 30
C[999] = 2997
```

The output was validated against the expected CPU-side calculations.

## 8. Relevance to the Main Project

The practical introduces fundamental CUDA concepts required for implementing GPU-based numerical algorithms.

The concepts of parallel indexing, kernel execution, GPU memory allocation, and host-device data transfer will be useful when implementing the Conjugate Gradient method for solving large sparse linear systems.

## 9. Conclusion

The introductory CUDA vector addition program was successfully implemented, compiled, and executed using a cloud-based NVIDIA GPU.

This practical provided hands-on experience with the CUDA programming model and established the foundation for subsequent GPU memory management, optimization, and numerical computing experiments.

## 10. Next Steps

- Study the GPU memory hierarchy.
- Understand global, shared, constant, and register memory.
- Investigate host-device memory transfer overhead.
- Develop the GPU implementation of the Conjugate Gradient method.
