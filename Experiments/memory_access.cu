%%writefile memory_access.cu

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err) << std::endl; \
            return 1;                                         \
        }                                                     \
    } while (0)

// Coalesced memory access
__global__ void coalescedAccess(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}

// Non-coalesced memory access
__global__ void nonCoalescedAccess(
    const float* input,
    float* output,
    int n,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int index = i * stride;
        output[index] = input[index] * 2.0f;
    }
}

int main() {

    const int n = 1 << 20;
    const int stride = 32;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int totalElements = n * stride;

    const size_t bytes = (size_t)totalElements * sizeof(float);

    float *d_input, *d_output;

    // Allocate enough memory for strided accesses
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));

    CUDA_CHECK(cudaMemset(d_input, 0, bytes));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int repetitions = 100;

    // Warm-up both kernels
    coalescedAccess<<<blocks, threads>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());

    nonCoalescedAccess<<<blocks, threads>>>(
        d_input, d_output, n, stride
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure coalesced access
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repetitions; i++) {
        coalescedAccess<<<blocks, threads>>>(
            d_input, d_output, n
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float coalescedTime;
    CUDA_CHECK(cudaEventElapsedTime(
        &coalescedTime, start, stop
    ));

    // Measure non-coalesced access
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repetitions; i++) {
        nonCoalescedAccess<<<blocks, threads>>>(
            d_input, d_output, n, stride
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float nonCoalescedTime;
    CUDA_CHECK(cudaEventElapsedTime(
        &nonCoalescedTime, start, stop
    ));

    // Average execution time per kernel launch
    coalescedTime /= repetitions;
    nonCoalescedTime /= repetitions;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\nMemory Access Experiment\n";
    std::cout << "------------------------\n";
    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Elements processed: " << n << '\n';
    std::cout << "Stride: " << stride << '\n';
    std::cout << "Repetitions: " << repetitions << '\n';

    std::cout << "Coalesced access: "
              << coalescedTime << " ms\n";

    std::cout << "Non-coalesced access: "
              << nonCoalescedTime << " ms\n";

    std::cout << "Time ratio: "
              << nonCoalescedTime / coalescedTime
              << "x\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
