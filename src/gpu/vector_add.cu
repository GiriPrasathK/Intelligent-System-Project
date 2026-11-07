%%writefile vector_add.cu

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                        \
                      << cudaGetErrorString(err) << std::endl; \
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
    const int n = 1000;
    const int threadsPerBlock = 256;
    const int blocks =
        (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> A(n), B(n), C(n);

    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        if (C[i] != A[i] + B[i]) {
            std::cerr << "Validation failed at index "
                      << i << std::endl;
            return 1;
        }
    }

    std::cout << "Vector addition: PASS\n";
    std::cout << "Elements: " << n << '\n';
    std::cout << "Threads per block: " << threadsPerBlock << '\n';
    std::cout << "Blocks: " << blocks << '\n';
    std::cout << "C[0] = " << C[0] << '\n';
    std::cout << "C[10] = " << C[10] << '\n';
    std::cout << "C[999] = " << C[999] << '\n';

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
