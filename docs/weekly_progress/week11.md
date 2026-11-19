    # Week 11 – Scalability Analysis

## Objective
Evaluate the scalability of the Conjugate Gradient (CG) method across CPU, local GPU (GTX 1080), and cloud GPU (Tesla T4).

## Hardware Configuration

- **CPU:** 2 × Intel Xeon E5-2667 v4, 16 cores, 32 threads
- **RAM:** 128 GB
- **Local GPU:** NVIDIA GTX 1080 (8 GB VRAM)
- **Cloud GPU:** NVIDIA Tesla T4

## Benchmark Results

Mean execution time in milliseconds:

| Matrix Size | CPU | GTX 1080 | Tesla T4 |
|---:|---:|---:|---:|
| 1,000 | 55.96 | 894.39 | 487.15 |
| 5,000 | 463.78 | 5369.96 | 1136.93 |
| 10,000 | 1350.79 | 10774.02 | 2635.40 |
| 20,000 | 25376.92 | 21750.59 | 4847.70 |
| 50,000 | 75232.93 | 53117.17 | 12149.62 |

## Key Observations

- CPU performs better for smaller matrices.
- Both GPUs outperform CPU at matrix sizes 20,000 and 50,000.
- GPU crossover occurs between 10,000 and 20,000.
- At 50,000, GTX 1080 achieves **1.42× speedup**.
- Tesla T4 achieves **6.19× speedup** over CPU.

## Resource Utilization

Monitoring during the 50,000 matrix benchmark:

- **CPU:** 7.41% average and 18.35% peak total-system utilization.
- **GTX 1080:** 39% peak GPU utilization.
- **GPU memory:** 1,197 MiB peak usage.
- **GPU temperature:** 31–33 °C.

## Generated Plots

1. `execution_time_scalability.png`
2. `execution_time_scalability_log.png`
3. `gpu_speedup_scalability.png`

## Conclusion

The experiments demonstrate that GPU acceleration becomes beneficial for larger matrix sizes. The Tesla T4 achieved the best performance, providing a 6.19× speedup over CPU at matrix size 50,000.

Week 11 successfully evaluated scalability, compared hardware performance, monitored resource utilization, and generated visualizations.