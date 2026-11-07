%%writefile memory_transfer.cu

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << cudaGetErrorString(err) << std::endl; \
            return 1;                                         \
        }                                                     \
    } while (0)

__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {

    const int n = 1000000;
    const size_t bytes = n * sizeof(float);

    const int threadsPerBlock = 256;
    const int blocks =
        (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> A(n), B(n), C(n);

    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure Host-to-Device transfer
    cudaEventRecord(start);

    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), bytes, cudaMemcpyHostToDevice
    ));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float h2d_ms;
    cudaEventElapsedTime(&h2d_ms, start, stop);

    // Measure GPU kernel execution
    cudaEventRecord(start);

    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // Measure Device-to-Host transfer
    cudaEventRecord(start);

    CUDA_CHECK(cudaMemcpy(
        C.data(), d_C, bytes, cudaMemcpyDeviceToHost
    ));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float d2h_ms;
    cudaEventElapsedTime(&d2h_ms, start, stop);

    // Measure complete wall-clock time
    auto total_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), bytes, cudaMemcpyHostToDevice
    ));

    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        C.data(), d_C, bytes, cudaMemcpyDeviceToHost
    ));

    auto total_end = std::chrono::high_resolution_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(
            total_end - total_start
        ).count();

    // Validate output
    bool valid = true;

    for (int i = 0; i < n; i++) {
        if (C[i] != A[i] + B[i]) {
            valid = false;
            break;
        }
    }

    std::cout << "\nMemory Transfer Experiment\n";
    std::cout << "--------------------------\n";

    std::cout << "GPU: ";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << prop.name << '\n';

    std::cout << "Elements: " << n << '\n';
    std::cout << "H2D Transfer: " << h2d_ms << " ms\n";
    std::cout << "Kernel Execution: " << kernel_ms << " ms\n";
    std::cout << "D2H Transfer: " << d2h_ms << " ms\n";
    std::cout << "Total Wall Time: " << total_ms << " ms\n";
    std::cout << "Validation: " << (valid ? "PASS" : "FAIL") << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return valid ? 0 : 1;
}
