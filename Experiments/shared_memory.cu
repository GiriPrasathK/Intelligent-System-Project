%%writefile shared_memory.cu

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define N (1 << 20)
#define THREADS 256
#define REPETITIONS 100

// Check CUDA errors
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err) << std::endl; \
            return 1;                                         \
        }                                                     \
    } while (0)

// Kernel 1: Reduction using global memory
__global__ void globalReduction(
    const float* input,
    float* scratch,
    float* partial
) {
    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x;
    int idx = base + tid;

    // Copy input into global scratch memory
    scratch[idx] = input[idx];

    __syncthreads();

    // Perform reduction using global memory
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            scratch[base + tid] +=
                scratch[base + tid + stride];
        }

        __syncthreads();
    }

    // Store block result
    if (tid == 0) {
        partial[blockIdx.x] = scratch[base];
    }
}

// Kernel 2: Reduction using shared memory
__global__ void sharedReduction(
    const float* input,
    float* partial
) {
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    sdata[tid] = input[idx];

    __syncthreads();

    // Perform reduction using shared memory
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // Store block result
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// Measure average kernel execution time
template <typename Kernel, typename... Args>
float measureKernel(Kernel kernel, Args... args) {
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    kernel<<<(N / THREADS), THREADS>>>(args...);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int i = 0; i < REPETITIONS; i++) {
        kernel<<<(N / THREADS), THREADS>>>(args...);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalTime = 0.0f;
    cudaEventElapsedTime(&totalTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return totalTime / REPETITIONS;
}

int main() {
    const int blocks = N / THREADS;

    size_t inputBytes = N * sizeof(float);
    size_t scratchBytes = N * sizeof(float);
    size_t partialBytes = blocks * sizeof(float);

    float *h_input, *h_partial;
    float *d_input, *d_scratch, *d_partial;

    h_input = new float[N];
    h_partial = new float[blocks];

    // Initialize input with 1.0
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_scratch, scratchBytes));
    CUDA_CHECK(cudaMalloc(&d_partial, partialBytes));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input, inputBytes,
        cudaMemcpyHostToDevice
    ));

    // Measure global memory reduction
    float globalTime = measureKernel(
        globalReduction, d_input, d_scratch, d_partial
    );

    CUDA_CHECK(cudaMemcpy(
        h_partial, d_partial, partialBytes,
        cudaMemcpyDeviceToHost
    ));

    double globalSum = 0.0;

    for (int i = 0; i < blocks; i++) {
        globalSum += h_partial[i];
    }

    // Measure shared memory reduction
    float sharedTime = measureKernel(
        sharedReduction, d_input, d_partial
    );

    CUDA_CHECK(cudaMemcpy(
        h_partial, d_partial, partialBytes,
        cudaMemcpyDeviceToHost
    ));

    double sharedSum = 0.0;

    for (int i = 0; i < blocks; i++) {
        sharedSum += h_partial[i];
    }

    // Display results
    std::cout << "\nCUDA Shared Memory Experiment\n";
    std::cout << "-----------------------------\n";

    std::cout << "Elements: " << N << "\n";
    std::cout << "Threads per block: " << THREADS << "\n";
    std::cout << "Number of blocks: " << blocks << "\n";

    std::cout << "\nGlobal Memory Reduction:\n";
    std::cout << "Sum: " << globalSum << "\n";
    std::cout << "Average time: " << globalTime << " ms\n";

    std::cout << "\nShared Memory Reduction:\n";
    std::cout << "Sum: " << sharedSum << "\n";
    std::cout << "Average time: " << sharedTime << " ms\n";

    std::cout << "\nExpected Sum: " << N << "\n";

    if (std::abs(globalSum - N) < 0.01 &&
        std::abs(sharedSum - N) < 0.01) {
        std::cout << "\nValidation: PASS\n";
    } else {
        std::cout << "\nValidation: FAIL\n";
    }

    if (sharedTime > 0.0f) {
        std::cout << "Global / Shared Time Ratio: "
                  << globalTime / sharedTime << "x\n";
    }

    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_partial));

    delete[] h_input;
    delete[] h_partial;

    return 0;
}
