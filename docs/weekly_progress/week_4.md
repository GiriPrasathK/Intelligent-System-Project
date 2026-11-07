# Week 4: GPU Memory Hierarchy and Memory Management

## 1. Objective

Understand the GPU memory hierarchy, host-device memory transfers, memory access patterns, and synchronization. Investigate how memory transfer overhead affects GPU application performance.

## 2. Concepts Studied

- GPU registers
- Shared memory
- Global memory
- Constant memory
- Host-to-device memory transfer
- Device-to-host memory transfer
- Memory coalescing
- CUDA synchronization

## 3. Practical Exercise: Memory Transfer and Kernel Timing

**Program:** CUDA Vector Addition with Memory Transfer Timing

The vector addition program was extended to measure host-to-device transfer time, GPU kernel execution time, device-to-host transfer time, and total wall-clock execution time.

The experiment used CUDA events to measure GPU operations and a host-side high-resolution clock to measure the complete execution.

## 4. Experimental Environment

- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- CUDA Toolkit: 13.0
- Programming language: CUDA C++
- Number of elements: 1,000,000
- Data type: float (32-bit)
- Threads per block: 256
- Number of blocks: 3,907

## 5. Experimental Results

| Measurement | Execution Time |
|---|---:|
| Host-to-Device Transfer | 2.64141 ms |
| GPU Kernel Execution | 0.146976 ms |
| Device-to-Host Transfer | 0.987552 ms |
| Total Wall Time | 2.90553 ms |

The program successfully completed execution and passed output validation.

## 6. Analysis

The experiment demonstrates that memory transfer overhead can be significant compared with GPU computation time.

The host-to-device transfer involved two input vectors, whereas the device-to-host transfer involved one output vector. This explains why the measured H2D transfer time was higher in this experiment.

The GPU kernel performed vector addition in approximately 0.147 ms. The measured transfer times demonstrate the importance of considering data movement when evaluating GPU application performance.

The individual transfer and kernel measurements were collected separately from the total wall-clock measurement. Therefore, their sum is not expected to exactly match the total wall-clock time.

These results are preliminary observations from a single run and require repeated measurements for reliable performance analysis.

## 7. Conclusion

The practical provided hands-on experience with CUDA events, host-device memory transfers, GPU kernel execution, and execution-time measurement.

It also demonstrated why memory transfer overhead must be considered when developing and evaluating GPU-accelerated numerical algorithms such as the Conjugate Gradient method.

## 8. Next Steps

- Study coalesced and non-coalesced memory access.
- Implement a CUDA shared-memory example.
- Investigate synchronization between GPU threads.
- Explore memory optimization techniques for sparse matrix-vector multiplication.
