Week 10 – Docker Containerization and GPU Execution
1. Objective
The objective of Week 10 was to containerize the Conjugate Gradient (CG) implementation using Docker and verify that GPU computing works inside the Docker container.

This makes the project easier to reproduce across different environments without manually installing all dependencies.
2. Environment and Tools
The following environment and tools were used:
Component	Specification
Operating System	Windows 10 Pro
Linux Environment	Ubuntu on WSL 2
Container Platform	Docker Desktop
Docker Engine	29.9.3
Base Docker Image	nvidia/cuda:12.2.2-devel-ubuntu22.04
CUDA Version	12.2
GPU	NVIDIA GeForce GTX 1080 (8 GB VRAM)
Python	Python 3.10.12 (inside container)
GPU Programming	Numba CUDA
Scientific Libraries	NumPy, SciPy, Numba, Pandas, Matplotlib

3. Files Added
The following files were created for Docker containerization:
File	Description
Dockerfile	Defines the Docker image and installation steps.
requirements.txt	Lists the Python dependencies required by the project.
.dockerignore	Excludes unnecessary files from the Docker build context.
src/cg_docker.py	Contains the CPU and GPU Conjugate Gradient implementations used for Docker testing.

4. Docker Configuration
Dockerfile
The Docker image was created using the NVIDIA CUDA development image.
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


Python Dependencies
The requirements.txt file contains:
numpy
scipy
numba
pandas
matplotlib


Docker Image Build
The Docker image was built using:
docker build -t cg-project:cuda .


The build completed successfully.
5. GPU Access Verification
GPU access from Docker was verified using:
docker run --rm --gpus all cg-project:cuda nvidia-smi


The container successfully detected the NVIDIA GeForce GTX 1080 GPU.

The Python CUDA environment was also tested:
docker run --rm --gpus all cg-project:cuda python3 -c "import numpy, scipy, numba; from numba import cuda; print('NumPy:', numpy.__version__); print('SciPy:', scipy.__version__); print('Numba:', numba.__version__); print('CUDA available:', cuda.is_available()); print('GPU:', cuda.get_current_device().name)"


The output confirmed:
- NumPy was successfully imported.
- SciPy was successfully imported.
- Numba was successfully imported.
- CUDA was available inside the container.
- The NVIDIA GeForce GTX 1080 was detected.
A CUDA vector addition test was also executed successfully, confirming that CUDA kernels could run inside the Docker container.
6. Conjugate Gradient Implementation
A separate script, src/cg_docker.py, was created to test the CPU and GPU implementations inside Docker.
Test Matrix
A sparse symmetric positive-definite tridiagonal matrix was generated.

The matrix size was:
\[
N = 1000
\]

The matrix was constructed as:
\[
A =
\begin{bmatrix}
2 & -1 & 0 & \cdots & 0\\
-1 & 2 & -1 & \cdots & 0\\
0 & -1 & 2 & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & -1\\
0 & 0 & 0 & -1 & 2
\end{bmatrix}
\]

The matrix contains 2,998 non-zero elements.
Execution
The following operations were performed:
1. Generated the sparse matrix and right-hand-side vector.
2. Executed the CPU Conjugate Gradient implementation using SciPy.
3. Executed the GPU Conjugate Gradient implementation using Numba CUDA.
4. Measured execution time for both implementations.
5. Calculated the residual and compared the solutions.
6. Validated the numerical correctness of both implementations.
7. Experimental Results
The following results were obtained from the Docker execution.
CPU Execution
Metric	Result
Convergence Status	0
Execution Time	0.01844 seconds
True Residual	\(3.76 \times 10^{-12}\)
Validation	Passed

GPU Execution
Metric	Result
Iterations	500
Execution Time	1.97998 seconds
Returned Residual	\(5.19 \times 10^{-13}\)
True Residual	\(5.19 \times 10^{-13}\)
Validation	Passed

CPU vs GPU Comparison
Metric	Result
CPU Solution Error	\(1.05 \times 10^{-12}\)
GPU Solution Error	\(1.13 \times 10^{-12}\)
CPU-GPU Solution Difference	\(1.23 \times 10^{-12}\)
GPU Speedup	0.0093×

8. Observations
- Both CPU and GPU implementations successfully solved the system.
- The residual values were very small, confirming numerical correctness.
- The CPU implementation was faster for this particular test case.
- The GPU implementation took approximately 1.98 seconds, compared to approximately 0.018 seconds for the CPU.
- Numba generated a performance warning indicating that the grid size was small and could result in GPU under-utilization.
The GPU's lower performance for this test is expected because the matrix size is relatively small. Kernel launch overhead, synchronization, and data transfer overhead can dominate execution time for smaller workloads.

This result does not contradict the earlier large-scale benchmark results, where the GPU implementations showed better performance for larger matrices.
9. GitHub Integration
The Docker-related files were committed and pushed to the project repository.

Repository: Intelligent-System-Project

Commit message:
Add Docker containerization for CG implementation


Commit hash:
5d0a3af


The following files were included in the commit:
- Dockerfile
- requirements.txt
- .dockerignore
- src/cg_docker.py
10. Conclusion
Week 10 successfully completed the Docker containerization of the Conjugate Gradient project.

The project was packaged into a Docker image with the required Python dependencies and CUDA environment. GPU passthrough was verified, and CUDA kernels were successfully executed inside the container.

The CPU and GPU Conjugate Gradient implementations were also tested within Docker, and both produced numerically valid solutions.

This establishes a reproducible execution environment for the project and provides a foundation for further scalability testing, performance analysis, and final integration.
