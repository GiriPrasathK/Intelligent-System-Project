# Week 8 – Cloud Infrastructure and Virtualization

## 1. Objective

Set up the required cloud environment and deploy a basic application using the selected cloud platform.

## 2. Selected Cloud Environment

Google Colab Free was selected as the cloud environment.

The runtime was configured and verified with the following resources:

| Component | Configuration |
|---|---|
| Platform | Google Colab Free |
| Operating System | Linux |
| Python | 3.13.18 |
| CPU Cores | 2 |
| System RAM | 12.67 GB |
| Runtime Storage | 107.72 GB |
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 14.56 GB |
| PyTorch | 2.14.1+cu130 |
| CUDA Availability | True |

## 3. Application Development

A GPU Cloud Resource Monitoring Application was developed using Python, PyTorch and Gradio.

The application provides two main functionalities:

**Resource Monitoring**
- Displays CPU core count and utilization.
- Displays system RAM usage.
- Displays storage information.
- Detects the GPU and displays its memory capacity.
- Checks CUDA availability.

**GPU Benchmarking**
- Executes matrix multiplication using PyTorch.
- Performs computation directly on the NVIDIA GPU.
- Measures execution time.
- Verifies the output dimensions.

## 4. GPU Benchmark Results

| Parameter | Result |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Operation | Matrix Multiplication |
| Matrix A | 2000 × 2000 |
| Matrix B | 2000 × 2000 |
| Execution Time | 5.995 ms |
| Output Shape | (2000, 2000) |
| CUDA Status | Successful |

The successful execution confirms that the deployed application can perform GPU-accelerated computations using the cloud GPU.

*Note: The recorded 5.995 ms is the execution time reported for the submitted benchmark run, not a repeated-run average.*

## 5. Technologies Used

- Google Colab – Cloud computing environment
- Python – Application development
- PyTorch – GPU computation
- CUDA – GPU acceleration
- Gradio – Interactive web interface
- psutil – System resource monitoring

## 6. Conclusion

The cloud environment was successfully configured using Google Colab Free. A basic interactive GPU resource monitoring application was developed and deployed using Gradio. The application successfully detected the NVIDIA Tesla T4 GPU and executed matrix multiplication using CUDA-enabled PyTorch.

This practical establishes the cloud infrastructure required for subsequent deployment of the Conjugate Gradient project.

**Week 8 Status: Completed – Basic cloud application successfully deployed and tested.**
