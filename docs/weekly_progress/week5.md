# WEEK 5 – GPU PERFORMANCE ANALYSIS AND PROFILING

**Course:** GPU Computing and Cloud Computing  
**Project:** Performance and Scalability Analysis of the Conjugate Gradient Method Across CPU, Local GPU, and Cloud GPU Environments

---

## 1. Introduction

The fifth week focused on understanding GPU performance analysis and profiling techniques. The objective was to evaluate GPU execution efficiency by studying performance metrics, memory coalescing, occupancy, CPU–GPU transfer overhead and kernel execution behavior.

Experiments were conducted using Google Colab with an NVIDIA Tesla T4 GPU. Sparse Matrix-Vector Multiplication (SpMV) using the Compressed Sparse Row (CSR) format was selected as the primary workload for performance evaluation.

NVIDIA Nsight Compute was used to profile the CUDA kernel and investigate its memory utilization, instruction execution, occupancy and warp scheduling behavior.

## 2. Objectives

The main objectives of Week 5 were:

1. Understand GPU performance metrics such as execution time, speedup, throughput and occupancy.
2. Study memory coalescing and its impact on GPU performance.
3. Understand GPU occupancy and resource utilization.
4. Analyze CPU–GPU data transfer overhead.
5. Compare CPU and GPU execution performance.
6. Profile CUDA kernels using NVIDIA Nsight Compute.
7. Identify potential performance bottlenecks through memory and instruction-level analysis.

## 3. Concepts Studied

### 3.1 GPU Performance Metrics

GPU performance was evaluated using execution time, speedup, throughput, memory bandwidth and occupancy.

### 3.2 Memory Coalescing

Memory coalescing refers to combining memory requests from threads within a warp when they access nearby memory locations. Efficient memory access patterns can reduce unnecessary memory transactions and improve GPU performance.

### 3.3 GPU Occupancy

Occupancy is influenced by register usage, shared memory requirements and thread-block configuration.

### 3.4 CPU–GPU Execution Overhead

GPU applications frequently require data transfers between host memory and device memory.

### 3.5 GPU Profiling

NVIDIA Nsight Compute was introduced as a GPU kernel profiling tool. It provides information about kernel execution, memory throughput, occupancy, instruction statistics and warp scheduling.

## 4. Experimental Environment

The experiments were conducted using the following configuration:

| Component | Specification |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU Architecture | Turing |
| Compute Capability | 7.5 |
| CUDA Toolkit | 13.0 |
| Nsight Compute | 2025.3.1.0 |
| Programming Language | Python |
| CUDA Programming | Numba CUDA |
| Sparse Matrix Format | CSR |
| Data Type | float64 |

## 5. Practical Experiments

### 5.1 Practical 1 – CPU vs GPU SpMV Performance Comparison

**Objective:** Compare the execution performance of sparse matrix-vector multiplication on CPU and GPU.

A tridiagonal sparse matrix was generated using SciPy and represented in CSR format. The matrix was transferred to the GPU, where a custom CUDA kernel performed matrix-vector multiplication.

**Experimental configuration:**

- Matrix size: 100,000 × 100,000
- Number of non-zero elements: 299,998
- Data type: float64
- GPU threads per block: 256
- Number of blocks: 391

**Results:**

| Execution Environment | Execution Time |
|---|---:|
| CPU (SciPy) | 0.503167 ms |
| GPU (Tesla T4) | 0.137187 ms |


**Observation:**

The GPU achieved approximately 3.67 times faster execution than the CPU for the tested workload. The GPU timing represents kernel execution time and excludes host-device data transfer overhead.

### 5.2 Practical 2 – Effective Memory Bandwidth Estimation

**Objective:** Estimate the effective memory bandwidth of the CSR SpMV kernel.

The estimated data traffic included matrix values, column indices, row pointers, input vector and output vector.


**Results:**

| Parameter | Value |
|---|---:|
| Estimated data traffic | 5.34 MB |
| GPU execution time | 0.137187 ms |
| Estimated effective bandwidth | 40.82 GB/s |

**Observation:**

The estimated effective bandwidth was approximately 40.82 GB/s. This represents an algorithmic estimate of data traffic and does not directly measure physical DRAM traffic, since caching and actual memory transactions can change the amount of data transferred.

### 5.3 Practical 3 – CUDA Block-Size Comparison

**Objective:** Investigate the effect of thread-block size on CUDA kernel execution performance.

Four different thread-block configurations were evaluated using the same CSR SpMV workload.

**Results:**

| Threads per Block | Number of Blocks | Execution Time |
|---:|---:|---:|
| 64 | 1563 | 0.155725 ms |
| 128 | 782 | 0.119974 ms |
| 256 | 391 | 0.162763 ms |
| 512 | 196 | 0.099898 ms |

**Observation:**

Among the tested configurations, 512 threads per block produced the lowest execution time of 0.099898 ms in the second benchmark run.

The results demonstrate that thread-block configuration can influence GPU performance. However, the optimal configuration depends on the workload and hardware, and further repeated experiments are necessary to establish a reliable optimum.

### 5.4 Practical 4 – Basic GPU Kernel Profiling Using Nsight Compute

**Objective:** Profile the CUDA CSR SpMV kernel and examine GPU execution characteristics.

NVIDIA Nsight Compute was used to collect kernel-level performance metrics.

The following command was used:

```bash
ncu --set basic \
    --kernel-name-base function \
    --kernel-name regex:csr_spmv.* \
    --target-processes all \
    python spmv_profile.py
```

**Results:**

| Metric | Value |
|---|---:|
| Kernel duration | 32.03 µs |
| DRAM throughput utilization | 74.65% |
| Compute throughput | 25.74% |
| L1/TEX cache throughput | 37.13% |
| L2 cache throughput | 26.69% |
| Theoretical occupancy | 100% |
| Achieved occupancy | 98.12% |
| Registers per thread | 33 |
| Block size | 256 |
| Grid size | 391 |

**Observation:**

The kernel achieved approximately 98.12% occupancy. Memory throughput utilization was higher than compute throughput utilization, indicating that memory behavior is an important factor in the performance of the SpMV operation.

### 5.5 Practical 5 – Memory Workload Analysis

**Objective:** Investigate GPU memory utilization and cache behavior during SpMV execution.

The Memory Workload Analysis section of Nsight Compute was used.

**Results:**

| Metric | Value |
|---|---:|
| Memory throughput | 213.98 GB/s |
| Memory busy | 27.73% |
| Maximum bandwidth utilization | 72.40% |
| L1/TEX cache hit rate | 51.90% |
| L2 cache hit rate | 36.93% |
| Memory pipes busy | 12.22% |
| Local memory spilling requests | 0 |

**Observation:**

The kernel exhibited substantial memory activity, with a measured memory throughput of approximately 213.98 GB/s. The L1/TEX and L2 cache hit rates were approximately 51.90% and 36.93%, respectively.

No local memory spilling requests were recorded.

These results indicate that memory access behavior and cache utilization are important areas for further optimization.

### 5.6 Practical 6 – Instruction Statistics

**Objective:** Analyze the instruction execution behavior of the CUDA SpMV kernel.

The Instruction Statistics section of Nsight Compute was used to collect instruction-related metrics.

**Results:**

| Metric | Value |
|---|---:|
| Executed instructions | 568,776 |
| Issued instructions | Approximately 576,017 |
| Average executed instructions per scheduler | 3,554.85 |
| Average issued instructions per scheduler | Approximately 3,600.11 |

**Observation:**

The two profiling passes produced nearly identical instruction counts. The results provide an overview of instruction execution across the GPU schedulers.

Further instruction-level analysis can be used to investigate floating-point, integer and memory instruction distributions.

### 5.7 Practical 7 – Warp State and Scheduler Statistics

**Objective:** Analyze GPU warp scheduling behavior and identify potential execution stalls.

Both Warp State Statistics and Scheduler Statistics sections of Nsight Compute were used.

**Warp State Statistics:**

| Metric | Observed Value |
|---|---:|
| Warp cycles per issued instruction | 29.45–30.07 |
| Warp cycles per executed instruction | 29.82–30.46 |
| Average active threads per warp | 32 |
| Average non-predicated-off threads per warp | 30.59 |
| L1TEX dependency stall share | Approximately 65–67% |

**Scheduler Statistics:**

| Metric | Pass 1 | Pass 2 |
|---|---:|---:|
| One or more eligible | 24.38% | 25.97% |
| No eligible | 75.62% | 74.03% |
| Issued warps per scheduler | 0.24 | 0.26 |
| Active warps per scheduler | 7.01 | 6.91 |
| Eligible warps per scheduler | 0.46 | 0.48 |

**Observation:**

The profiling results indicate significant memory dependency stalls. Although the kernel achieved high occupancy, a considerable proportion of active warps were not eligible to issue their next instruction.

The L1TEX dependency stalls suggest that memory access latency is an important factor affecting kernel execution. Improving memory access patterns, coalescing and data locality may help improve performance.

### 5.8 Practical 8 – CPU–GPU Data Transfer Overhead

**Objective:** Measure the time required for data transfer between CPU and GPU and compare it with GPU computation time.

Vector addition was performed using one million float64 elements on the NVIDIA Tesla T4.

**Results:**

| Operation | Execution Time | Percentage |
|---|---:|---:|
| Host to Device (H2D) | 8.9148 ms | 46.63% |
| GPU kernel execution | 0.4165 ms | 2.18% |
| Device to Host (D2H) | 9.7867 ms | 51.19% |
| Total measured time | 19.1180 ms | 100% |

**Observation:**

Data transfers accounted for approximately 97.82% of the combined measured stage times, while GPU computation accounted for only 2.18%.

This experiment demonstrates that data transfer overhead can significantly affect overall GPU application performance, especially for operations involving relatively little computation.

The experiment also highlights the importance of keeping intermediate data on the GPU whenever possible.

## 6. Overall Experimental Results

The following table summarizes the major findings from Week 5.

| Experiment | Major Result |
|---|---|
| CPU vs GPU SpMV | 3.67× kernel-only speedup |
| Effective bandwidth estimation | 40.82 GB/s |
| Block-size comparison | 512 threads fastest in the tested run |
| Basic kernel profiling | 98.12% achieved occupancy |
| Memory workload analysis | 213.98 GB/s profiler throughput |
| Instruction statistics | 568,776 executed instructions |
| Warp state analysis | Approximately 65–67% L1TEX dependency stall share |
| Scheduler analysis | Approximately 74–76% cycles with no eligible warp |
| Transfer overhead | 97.82% of measured stage time spent transferring data |

## 7. Key Observations and Findings

The experiments conducted during Week 5 resulted in the following findings:

1. The GPU demonstrated approximately 3.67× faster kernel execution than the CPU for the tested SpMV workload.
2. Thread-block configuration influenced execution performance, with 512 threads per block providing the best result among the tested configurations in the second run.
3. The SpMV kernel achieved high occupancy of approximately 98.12%.
4. Memory throughput utilization was higher than compute throughput utilization.
5. Memory profiling indicated moderate L1 and L2 cache hit rates.
6. Warp State Statistics identified significant L1TEX memory dependency stalls.
7. Scheduler Statistics showed that many active warps were not eligible to issue instructions.
8. CPU–GPU transfer experiments demonstrated that data movement can dominate execution time when the computational workload is relatively small.

## 8. Relevance to the Main Project

The Week 5 experiments provide an important foundation for the project titled "Performance and Scalability Analysis of the Conjugate Gradient Method Across CPU, Local GPU, and Cloud GPU Environments."

The profiling results identified several potential optimization directions for the GPU implementation of the Conjugate Gradient algorithm:

- Keeping intermediate vectors on the GPU to minimize repeated host-device transfers.
- Improving memory access patterns in sparse matrix-vector multiplication.
- Investigating warp-level parallelism for CSR operations.
- Evaluating different thread-block configurations.
- Reducing unnecessary CPU–GPU synchronization.
- Comparing memory throughput, occupancy and execution behavior across different GPU architectures.

The current profiling experiments were conducted on a standalone CSR SpMV kernel. Further profiling of the complete Conjugate Gradient implementation will be necessary to understand the performance of the entire iterative algorithm.

## 9. Conclusion

Week 5 focused on GPU performance analysis and profiling. Multiple experiments were conducted using the NVIDIA Tesla T4 GPU, including CPU–GPU SpMV comparison, effective bandwidth estimation, block-size tuning, memory workload analysis, instruction statistics, warp scheduling and data transfer overhead measurement.

NVIDIA Nsight Compute provided detailed insights into the execution behavior of the CUDA kernel. The experiments demonstrated that high GPU occupancy does not necessarily guarantee efficient instruction execution, particularly when memory dependencies limit warp scheduling.

The transfer overhead experiment further demonstrated the importance of minimizing data movement between CPU and GPU.

Overall, Week 5 established a performance-analysis foundation for optimizing the Conjugate Gradient implementation and comparing its behavior across cloud and local GPU environments.

## 10. Future Work

The following activities will be considered in subsequent weeks:

1. Study GPU-based numerical and linear algebra operations.
2. Investigate CUDA libraries for sparse matrix operations.
3. Optimize the Conjugate Gradient implementation.
4. Explore GPU-resident vector operations and reductions.
5. Repeat profiling experiments on the local NVIDIA RTX A6000.
6. Compare performance and scalability across CPU, local GPU and cloud GPU environments.
