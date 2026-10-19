# Week 3: Introduction to CUDA Programming

## Objective
Understand the CUDA programming model, kernels, thread indexing, host/device execution, and basic CUDA programming.

## Concepts Learned
- CUDA is NVIDIA's parallel computing platform and programming model.
- The host refers to the CPU side of a CUDA application.
- The device refers to the GPU.
- A CUDA kernel executes on the GPU using multiple threads.
- Thread indexing assigns work to individual data elements.
- Host-to-device and device-to-host copies transfer data between CPU and GPU memory.

## Practical Exercise
**Program:** Vector addition

The program calculates:

C[i] = A[i] + B[i]

It initializes input arrays on the CPU, allocates GPU memory, copies the input arrays to the GPU, launches the CUDA kernel, copies the output back, and validates the results.

## Thread Configuration
- Input elements: 1,000
- Threads per block: 256
- Blocks: 4
- Total thread positions launched: 1,024
- Valid elements processed: 1,000

A bounds check prevents the extra thread positions from accessing outside the arrays.

## Validation
Expected result: `Vector addition: PASS`

**Actual status:** Pending execution on a compatible CUDA environment until verified.

## Relevance to the CG Project
The exercise introduces kernel launches, parallel indexing, memory allocation, and data transfer. These concepts will be useful when implementing and benchmarking GPU operations relevant to the Conjugate Gradient method.

## Next Steps
Study the GPU memory hierarchy and host–device transfer overhead in Week 4.
