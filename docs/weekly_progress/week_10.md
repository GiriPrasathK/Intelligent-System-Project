

Week 10 Progress Report
Docker Containerization and GPU Execution of the Conjugate Gradient Method
Project: Performance and Scalability Analysis of the Conjugate Gradient Method Across CPU, Local GPU, and Cloud GPU Environments
Week: 10
Environment: Ubuntu on WSL 2, Docker Desktop, NVIDIA GeForce GTX 1080
Repository: https://github.com/GiriPrasathK/Intelligent-System-Project
Commit: 5d0a3af — Add Docker containerization for CG implementation

1. Objective
The objective for Week 10 was to containerize the Conjugate Gradient (CG) implementation using Docker and verify that the container could access and execute workloads on the host machine's NVIDIA GPU.

Containerization makes the project easier to reproduce across compatible systems by packaging the required Python environment, dependencies, and execution instructions into a Docker image.

2. Work Completed
The following tasks were completed during this week:

Installed and configured Docker Desktop with the WSL 2 backend.
Enabled Docker integration for the Ubuntu WSL distribution.
Verified that Docker could run containers successfully.
Configured NVIDIA GPU access from Docker containers.
Created a requirements.txt file containing the Python dependencies.
Created a Dockerfile based on the NVIDIA CUDA 12.2.2 development image.
Created a .dockerignore file to exclude unnecessary files from the build context.
Implemented a Docker-compatible CG demonstration in src/cg_docker.py.
Built the Docker image and executed the CG program inside a GPU-enabled container.
Compared the CPU and GPU solutions and checked their numerical correctness.
3. Docker Configuration
3.1 Dependencies
The requirements.txt file contains:

numpy
scipy
numba
pandas
matplotlib
3.2 Dockerfile
The Docker image uses NVIDIA's CUDA 12.2.2 development image with Ubuntu 22.04 as its base. Python, pip, development headers, and build tools are installed, followed by the project dependencies.

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "src/cg_docker.py"]
3.3 Docker ignore configuration
The .dockerignore file excludes Git metadata, editor configuration, Python cache files, virtual environments, and environment files from the Docker build context.

.git
.vscode
__pycache__
*.pyc
.ipynb_checkpoints
.env
.venv
venv
4. Building and Testing the Docker Image
The Docker image was built using:

docker build -t cg-project:cuda .
The build completed successfully.

GPU access from inside the container was verified using:

docker run --rm --gpus all cg-project:cuda nvidia-smi
The command successfully detected the host's NVIDIA GeForce GTX 1080 GPU.

The Python CUDA environment was also checked from inside the container. The test confirmed that NumPy, SciPy, and Numba were importable and that Numba detected the GPU.

Component	Verified result
Docker image	Built successfully
GPU passthrough	Successful
GPU detected in container	NVIDIA GeForce GTX 1080
NumPy	2.2.6
SciPy	1.15.3
Numba	0.68.1
Numba CUDA availability	True
A CUDA vector-add test was also executed successfully inside the container, confirming that CUDA kernels could run through the Docker GPU runtime.

5. Conjugate Gradient Demonstration
A Docker-compatible demonstration was implemented in src/cg_docker.py.

The program creates a sparse, symmetric positive-definite tridiagonal matrix of size 1000 × 1000:

A
=
[
2
−
1
0
⋯
0
−
1
2
−
1
⋯
0
0
−
1
2
⋱
0
⋮
⋮
⋱
⋱
−
1
0
0
0
−
1
2
]
A= 
​
  
2
−1
0
⋮
0
​
  
−1
2
−1
⋮
0
​
  
0
−1
2
⋱
0
​
  
⋯
⋯
⋱
⋱
−1
​
  
0
0
0
−1
2
​
  
​
 
The matrix contains 2,998 non-zero elements. The program runs the CPU implementation using SciPy and the GPU implementation using Numba CUDA kernels.

Both implementations use the same test problem and are checked for numerical correctness.

6. Experimental Results
The following results were obtained from the Docker execution.

6.1 CPU execution
Metric	Result
Matrix size	1000 × 1000
Convergence status	0
Execution time	0.0184369 seconds
True residual	3.7613 × 10⁻¹²
Validation	Passed
6.2 GPU execution
Metric	Result
Matrix size	1000 × 1000
Iterations	500
Execution time	1.9799843 seconds
Returned residual	5.1891 × 10⁻¹³
True residual	5.1891 × 10⁻¹³
Validation	Passed
6.3 CPU–GPU comparison
Metric	Result
CPU solution error	1.0518 × 10⁻¹²
GPU solution error	1.1321 × 10⁻¹²
Difference between CPU and GPU solutions	1.2265 × 10⁻¹²
GPU speedup	0.00931×
The CPU and GPU solutions are numerically close, and both passed validation. For this small workload, the GPU execution was slower than the CPU execution.

7. Discussion
The GPU was slower for the 1000 × 1000 test problem. This result is expected for a relatively small workload because GPU execution involves kernel-launch and synchronization overhead. These costs can outweigh the parallel-computation benefit when the amount of work is limited.

Numba also displayed a performance warning indicating that a grid size of four blocks could lead to low GPU occupancy. This warning is consistent with the small workload and does not indicate that the computation failed.

The result should not be interpreted as evidence that the GPU is always slower. The project's larger benchmark experiments showed that GPU performance depends on problem size and hardware. In particular, the cloud NVIDIA T4 achieved a speedup of approximately 6.06× over the CPU at matrix size 50K in the earlier benchmark results. The Docker run in this week was a containerization and correctness demonstration at a smaller size, not a replacement for those benchmark experiments.

8. Files Added
The following files were added for Docker support:

File	Purpose
Dockerfile	Defines the Docker image and execution environment
requirements.txt	Lists required Python packages
.dockerignore	Excludes unnecessary files from the build context
src/cg_docker.py	Runs the CPU and GPU CG demonstration inside Docker
9. Reproducibility Commands
From the project root directory, the Docker image can be rebuilt using:

docker build -t cg-project:cuda .
To execute the CG demonstration with GPU access:

docker run --rm --gpus all cg-project:cuda
To verify GPU visibility inside the container:

docker run --rm --gpus all cg-project:cuda nvidia-smi
10. Conclusion
Week 10 successfully completed the Docker containerization task. The project dependencies and CG demonstration were packaged into a Docker image, and GPU passthrough was verified using the NVIDIA GeForce GTX 1080. A CUDA vector-add test and the CPU/GPU CG demonstration both executed successfully inside the container.

The numerical validation confirmed that the CPU and GPU implementations produced closely matching solutions. The small test workload ran faster on the CPU because of GPU launch and synchronization overhead, illustrating why performance comparisons must consider problem size.

The Docker-related files were committed and pushed to the GitHub repository in commit 5d0a3af. This provides a reproducible container-based execution environment for the project's subsequent scalability analysis and final validation.





