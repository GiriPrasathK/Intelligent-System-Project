# Week 4: GPU Memory Hierarchy and Memory Optimization

## 1. Objectives

- Understand the GPU memory hierarchy.
- Study host-to-device and device-to-host memory transfers.
- Understand coalesced and non-coalesced memory access.
- Explore shared memory and thread synchronization.

## 2. GPU Memory Hierarchy

CUDA provides different types of memory:

- **Registers:** Fast, private storage for individual threads.
- **Shared Memory:** Fast on-chip memory shared among threads within a block.
- **Global Memory:** Large GPU memory accessible by all threads.
- **Constant Memory:** Read-only memory optimized for constant data access.

## 3. Practical 1: Host-Device Memory Transfer

**Objective:** Measure data transfer and kernel execution time.

**Environment:** Google Colab – NVIDIA Tesla T4.

**Configuration:**
- Elements: 1,000,000
- Data type: float32
- Threads per block: 256
- Blocks: 3,907

**Results:**

| Operation | Time (ms) |
|---|---:|
| Host-to-Device (H2D) | 2.64141 |
| Kernel execution | 0.146976 |
| Device-to-Host (D2H) | 0.987552 |
| Total wall time | 2.90553 |

**Observation:** Data transfer overhead can be significantly higher than kernel execution time.

**Validation:** PASS

## 4. Practical 2: Coalesced vs Non-Coalesced Memory Access

**Objective:** Compare GPU memory access patterns.

**Configuration:**
- Elements processed: 1,048,576
- Stride: 32
- Threads per block: 256
- Blocks: 4,096
- Repetitions: 100

**Results:**

| Access Pattern | Average Time (ms) |
|---|---:|
| Coalesced | 0.0393126 |
| Non-coalesced | 0.932594 |

**Observed time ratio:** 23.72×

**Observation:** Coalesced memory access was considerably faster in this experiment. However, the non-coalesced implementation accessed a larger memory footprint, so the difference cannot be attributed exclusively to memory coalescing.

## 5. Practical 3: Shared Memory and Synchronization

**Objective:** Compare parallel reduction using global memory and shared memory.

**Configuration:**
- Elements: 1,048,576
- Threads per block: 256
- Blocks: 4,096
- Repetitions: 100

**Results:**

| Metric | Global Memory | Shared Memory |
|---|---:|---:|
| Average time (ms) | 0.0642867 | 0.0644096 |
| Computed sum | 1,048,576 | 1,048,576 |

**Observation:** Both implementations produced the correct result. Their execution times were nearly identical. GPU caching and synchronization overhead may have influenced the measurements.

The `__syncthreads()` function was used to synchronize threads within each block during reduction.

## 6. Key Learnings

- GPU memory hierarchy affects application performance.
- Host-device data transfers introduce overhead.
- Coalesced memory access improves memory efficiency.
- Shared memory enables efficient communication between threads in a block.
- `__syncthreads()` ensures synchronization between threads.
- Performance improvements depend on workload characteristics and GPU architecture.

## 7. Conclusion

Week 4 provided practical experience with CUDA memory management, memory access patterns, shared memory, and synchronization. Three experiments were conducted on an NVIDIA Tesla T4 GPU, demonstrating the importance of memory behavior in GPU computing.
