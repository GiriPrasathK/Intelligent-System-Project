# Week 7 – Identification of Suitable Cloud Resources

## 1. Objective

Identify suitable cloud computing resources for the selected project, *Performance and Scalability Analysis of the Conjugate Gradient Method Across CPU, Local GPU, and Cloud GPU Environments.*

## 2. Project Requirements

The project requires a cloud environment capable of:

- Executing CPU- and GPU-based computations.
- Supporting NVIDIA CUDA programming.
- Running Python, Numba CUDA and PyTorch applications.
- Handling sparse matrices and numerical linear algebra operations.
- Storing experimental datasets, benchmark results and trained ML models.

## 3. Cloud Platform Evaluation

The following platforms were considered:

**Google Colab Free**
- Provides a cloud-hosted Python notebook environment.
- Offers access to GPU resources subject to availability and usage limits.
- Supports PyTorch, Numba and CUDA.
- Requires minimal environment configuration.

**Amazon EC2**
- Provides configurable GPU-enabled virtual machines.
- Supports CUDA-based applications.
- Uses a pay-as-you-go pricing model, requiring careful monitoring of running instances and associated storage costs.

**Google Cloud Compute Engine and Microsoft Azure**
- Provide configurable cloud virtual machines with GPU support.
- Offer greater control over computing resources but may involve additional configuration and costs.

## 4. Selected Cloud Platform

Google Colab Free was selected as the primary cloud platform for this project.

The selection was based on the following reasons:

1. It provides access to GPU computing without requiring a paid virtual machine.
2. It supports the programming frameworks used in the project.
3. It requires minimal infrastructure configuration.
4. The existing Conjugate Gradient implementation has already been successfully executed using an NVIDIA Tesla T4 GPU.
5. It is suitable for continuing the project's benchmarking and machine learning experiments.

## 5. Selected Computing Environment

| Component | Selected Resource |
|---|---|
| Cloud platform | Google Colab Free |
| GPU | NVIDIA Tesla T4 (when available) |
| Programming language | Python |
| GPU programming | Numba CUDA |
| ML framework | PyTorch |
| Storage | Google Drive and GitHub |
| Profiling | NVIDIA Nsight Compute, where available |

## 6. Limitations

Google Colab Free has certain limitations:

- GPU availability is not guaranteed.
- Runtime sessions may disconnect or reset.
- GPU usage is subject to quotas and restrictions.
- Runtime storage is temporary, requiring experimental results to be saved externally.

These limitations will be considered while conducting the remaining experiments.

## 7. Conclusion

Google Colab Free was identified as the most suitable cloud platform for the current project requirements. Its GPU support, compatibility with the existing implementation and lack of mandatory paid VM usage make it a practical choice for cloud-based CG benchmarking and GPU-accelerated machine learning experiments.

**Week 7 Status:** Cloud platform identified and selected. Detailed cloud deployment will be addressed in the subsequent project activities.
