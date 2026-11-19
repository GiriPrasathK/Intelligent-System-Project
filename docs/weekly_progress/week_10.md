# Week 10 – Containers for GPU Computing

**Course:** GPU Computing and Cloud Computing  
**Project:** Performance and Scalability Analysis of the Conjugate Gradient Method Across CPU, Local GPU, and Cloud GPU Environments  
**Duration:** October 20 – October 26, 2026  
**Status:** Completed

---

## 1. Objective

The objective of Week 10 was to understand Docker containerization and establish a reproducible environment for executing GPU-based applications.

As part of the project, the existing Conjugate Gradient (CG) implementation was containerized using Docker. GPU access from within the container was configured and tested to verify that CUDA-based computations could execute successfully.

## 2. Topics Covered

- Docker and containerization.
- Docker images and containers.
- GPU-enabled Docker containers.
- CUDA execution inside containers.
- Reproducible environments for GPU applications.

## 3. Practical Implementation

### 3.1 Docker Environment Setup

Docker Desktop was installed and configured with the WSL 2 backend. Ubuntu integration was enabled to execute Docker commands from the Linux environment.

The installation was verified using:

```bash
docker --version
docker run hello-world
```

Both commands executed successfully.

### 3.2 GPU Access Configuration

The NVIDIA CUDA development image was selected as the base image:

```text
nvidia/cuda:12.2.2-devel-ubuntu22.04
```

GPU passthrough was tested using:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

The container successfully detected the NVIDIA GeForce GTX 1080 GPU.

### 3.3 Project Containerization

The following files were created:

| File | Purpose |
|---|---|
| `Dockerfile` | Defines the Docker image configuration and installation steps. |
| `requirements.txt` | Specifies the required Python libraries. |
| `.dockerignore` | Excludes unnecessary files from the Docker build context. |
| `src/cg_docker.py` | Implements CPU and GPU Conjugate Gradient execution for Docker testing. |

The required Python libraries included NumPy, SciPy, Numba, Pandas, and Matplotlib.

The Docker image was built using:

```bash
docker build -t cg-project:cuda .
```

The build completed successfully.

### 3.4 CUDA Execution Inside Docker

The Python environment and CUDA availability were verified inside the container.

The following checks were performed:

- Imported NumPy, SciPy, and Numba successfully.
- Verified CUDA availability using Numba.
- Detected the NVIDIA GeForce GTX 1080.
- Executed a CUDA vector addition test successfully.

This confirmed that GPU computation was accessible from within the Docker container.

## 4. Conjugate Gradient Execution Inside Docker

A separate script, `src/cg_docker.py`, was developed to execute and compare the CPU and GPU implementations.

**Test configuration:**

| Parameter | Value |
|---|---|
| Matrix Size | 1000 × 1000 |
| Matrix Type | Sparse, symmetric positive-definite |
| Non-zero Elements | 2998 |
| GPU | NVIDIA GeForce GTX 1080 |
| GPU Programming Framework | Numba CUDA |

A tridiagonal sparse matrix was generated with diagonal elements equal to 2 and off-diagonal elements equal to -1.

Both CPU and GPU implementations were executed using the same test problem, and their execution times and numerical accuracy were evaluated.

### 4.1 CPU Results

| Metric | Result |
|---|---:|
| Convergence Status | 0 |
| Execution Time | 0.01844 seconds |
| True Residual | \(3.76 \times 10^{-12}\) |
| Validation | Passed |

### 4.2 GPU Results

| Metric | Result |
|---|---:|
| Number of Iterations | 500 |
| Execution Time | 1.97998 seconds |
| Returned Residual | \(5.19 \times 10^{-13}\) |
| True Residual | \(5.19 \times 10^{-13}\) |
| Validation | Passed |

### 4.3 CPU and GPU Comparison

| Metric | Result |
|---|---:|
| CPU Solution Error | \(1.05 \times 10^{-12}\) |
| GPU Solution Error | \(1.13 \times 10^{-12}\) |
| CPU-GPU Solution Difference | \(1.23 \times 10^{-12}\) |
| GPU Speedup | 0.0093× |

## 5. Observations

1. Both CPU and GPU implementations successfully solved the generated linear system.
2. The residual values were sufficiently small, confirming the numerical accuracy of both solutions.
3. The CPU implementation performed significantly faster for the test matrix of size 1000.
4. The GPU implementation experienced additional overhead from kernel launches, synchronization, and data transfers.
5. Numba generated a performance warning indicating possible GPU under-utilization because of the small grid size.

The results demonstrate that GPU acceleration does not necessarily improve performance for smaller workloads. The overhead associated with launching GPU kernels can outweigh the benefits of parallel execution.

This observation is consistent with the earlier benchmark results, where GPU performance improved for larger matrix sizes.

## 6. Overall Summary

During Week 10, the Conjugate Gradient project was successfully containerized using Docker.

The Docker environment was configured with the required Python dependencies and CUDA support. GPU passthrough was verified, and CUDA kernels were executed successfully inside the container.

The CPU and GPU implementations were also tested within Docker. Both produced numerically valid solutions, although the CPU performed better for the smaller test matrix.

The work establishes a reproducible environment for running the project and provides a foundation for further scalability and performance evaluation.

## 7. Conclusion

Week 10 successfully completed the containerization and GPU execution requirements. The project is now configured to run inside a Docker environment with NVIDIA GPU support.

This improves the reproducibility and portability of the implementation and provides a foundation for future scalability testing and performance evaluation.

**Next Week:** Week 11 – Distributed and Scalable Cloud Computing.
