// Author: Yash Deshpande
// Date  : 30-04-2026

#include<iostream>
#include<vector>
#include<utility>
#include <cuda.h>
#include <random>       // generate random numbers
#include <chrono>       // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>
#include <cstring>
#include <algorithm>   // std::swap
#include <cmath>
using namespace std::chrono;

#define BLOCK_DIM 1024

__global__ void matVec_kernel(float* matrix, float* b, float* b_next, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < rows) {
        float sum = 0.0f;
        for(int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * b[j];
        }
        b_next[i] = sum;
    }
}

// __global__ void reduce_kernel(float *b_next, float *partial_sums, unsigned int N) {
//     unsigned int segment = blockIdx.x * blockDim.x * 2;
//     unsigned int i = segment + threadIdx.x;

//     // Squared-sum
//     __shared__ float input_s[blockDim.x];
//     // load the shared memory with 1st iteration of sums
//     float v1 = (i               < N) ? b_next[i]               : 0.0f;
//     float v2 = (i + blockDim.x  < N) ? b_next[i + blockDim.x]  : 0.0f;
//     input_s[threadIdx.x] = v1*v1 + v2*v2;
//     __syncthreads();


//     for(unsigned int stride = blockDim.x/2; stride >= 1; stride/=2) {
//         if(threadIdx.x < stride) {
//             input_s[threadIdx.x] += input_s[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }

//     if(threadIdx.x == 0) {
//         partial_sums[blockIdx.x] = input_s[threadIdx.x];
//     }
// }

template <bool SquareInputs>
__global__ void reduce_kernel(const float*  input, float* output, unsigned int N)
{
    // Each block consumes 2 * blockDim.x elements
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i       = segment + threadIdx.x;

    __shared__ float s[BLOCK_DIM];
    // Bounds-checked load (predicated; no real divergence cost)
    float a = (i              < N) ? input[i]              : 0.0f;
    float b = (i + blockDim.x < N) ? input[i + blockDim.x] : 0.0f;

    // Compile-time switch: pass 0 squares, later passes just add.
    // The unused branch is removed by the compiler — zero runtime cost.
    if constexpr (SquareInputs) {
        s[threadIdx.x] = a*a + b*b;
    } else {
        s[threadIdx.x] = a + b;
    }
    __syncthreads();

    // Standard sequential-addressing tree reduction
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = s[0];
    }
}



// Reduce d_input[0..N-1] to a single squared sum, written to d_result[0].
//
// Caller supplies two device scratch buffers (d_scratchA, d_scratchB), each
// of size >= ceil(N / (2 * BLOCK_DIM)) floats. Passing them in lets you reuse
// them across power-iteration steps instead of paying cudaMalloc every loop.
void squaredSumGPU(const float* input_d,
                   unsigned int N,
                   float*       result_d,
                   float*       scratchA_d,
                   float*       scratchB_d) {
    const unsigned int elemsPerBlock = 2u * BLOCK_DIM;

    // ---------- Pass 0: square + first reduction ----------
    unsigned int blocks = (N + elemsPerBlock - 1) / elemsPerBlock;
    reduce_kernel<true><<<blocks, BLOCK_DIM>>>(input_d, scratchA_d, N);

    // ---------- Passes 1..K: keep reducing partial sums ----------
    float* readBuf  = scratchA_d;
    float* writeBuf = scratchB_d;
    unsigned int currentN = blocks;          // # partial sums to reduce next

    while (currentN > 1) {
        blocks = (currentN + elemsPerBlock - 1) / elemsPerBlock;
        reduce_kernel<false><<<blocks, BLOCK_DIM>>>(readBuf, writeBuf, currentN);
        std::swap(readBuf, writeBuf);        // ping-pong
        currentN = blocks;
    }

    // After the loop, readBuf[0] holds the final scalar.
    // Copy to a stable output location so the caller doesn't need to know
    // which scratch buffer ended up holding the result.
    cudaMemcpy(result_d, readBuf, sizeof(float), cudaMemcpyDeviceToDevice);
}

__global__ void normalizeDiffReduce_kernel(
    float* __restrict__       b_next,        // in-out: read raw, write normalized
    const float* __restrict__ b,             // in:    previous iteration's vector
    const float* __restrict__ d_normSq,      // in:    pointer to single float ||b_next||^2
    float* __restrict__       partial_sums,  // out:   per-block partial Σ|b_next' - b|
    unsigned int              N) {
    __shared__ float s[BLOCK_DIM];
    __shared__ float inv_norm;

    // Compute inv_norm once per block (broadcast via shared memory)
    if (threadIdx.x == 0) {
        // 1.0f / sqrtf(...) is more accurate; rsqrtf is the fast intrinsic.
        inv_norm = rsqrtf(d_normSq[0]);
    }
    __syncthreads();

    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i       = segment + threadIdx.x;
    unsigned int j       = i + blockDim.x;

    // Bounds-checked loads. b_next will be normalized in registers before write-back.
    float bn1 = (i < N) ? b_next[i] : 0.0f;
    float bn2 = (j < N) ? b_next[j] : 0.0f;
    float bp1 = (i < N) ? b[i]      : 0.0f;
    float bp2 = (j < N) ? b[j]      : 0.0f;

    bn1 *= inv_norm;
    bn2 *= inv_norm;

    // Write normalized values back in place (this is what the next matVec will read)
    if (i < N) b_next[i] = bn1;
    if (j < N) b_next[j] = bn2;

    // First reduction pass: each thread contributes two |diff| values
    s[threadIdx.x] = fabsf(bn1 - bp1) + fabsf(bn2 - bp2);
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = s[0];
    }
}

void normalizeDiffSumGPU(
    float*       d_b_next,
    const float* d_b,
    const float* d_normSq,     // single float, already computed
    float*       d_diffSum,    // single-float output: Σ|b_next' - b|
    float*       d_scratchA,
    float*       d_scratchB,
    unsigned int N) {
    const unsigned int eb = 2u * BLOCK_DIM;
    unsigned int blocks  = (N + eb - 1) / eb;

    // Pass 0: fused element-wise + first reduction
    normalizeDiffReduce_kernel<<<blocks, BLOCK_DIM>>>(
        d_b_next, d_b, d_normSq, d_scratchA, N);

    // Tail: standard sum-reduction of partial sums
    float *r = d_scratchA, *w = d_scratchB;
    unsigned int n = blocks;
    while (n > 1) {
        blocks = (n + eb - 1) / eb;
        reduce_kernel<false><<<blocks, BLOCK_DIM>>>(r, w, n);
        std::swap(r, w);
        n = blocks;
    }
    cudaMemcpy(d_diffSum, r, sizeof(float), cudaMemcpyDeviceToDevice);
}


void powerIteration_gpu(float* matrix, float* b, float* b_next, int rows, int cols, float* eigen_value, int maxIter = 100000, float tol = 1e-6) {
    int N = rows;
    // for timing purpose
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Allocate on the GPU
    float* matrix_d, *b_d, *b_next_d;

    cudaMalloc((void**) &matrix_d, rows * cols * sizeof(float));
    cudaMalloc((void**) &b_d, rows * sizeof(float));
    cudaMalloc((void**) &b_next_d, rows * sizeof(float));

    const unsigned int scratchElems = (N + 2u*BLOCK_DIM - 1u) / (2u*BLOCK_DIM);
    float *normSq_d, *diffSum_d, *scratchA_d, *scratchB_d;
    cudaMalloc(&normSq_d,    sizeof(float));
    cudaMalloc(&diffSum_d,   sizeof(float));
    cudaMalloc(&scratchA_d,  scratchElems * sizeof(float));
    cudaMalloc(&scratchB_d,  scratchElems * sizeof(float));

    // Copy on the GPU
    cudaMemcpy(matrix_d, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, rows *  sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_next_d, b_next, rows * sizeof(float), cudaMemcpyHostToDevice);

    
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBLocks = (N + numThreadsPerBlock - 1)/numThreadsPerBlock;

    cudaEventRecord(start);
    // Calculate on the GPU
    for (int iter = 0; iter < maxIter; ++iter) {

        // (1) b_next = A * b
        matVec_kernel<<<numBLocks, BLOCK_DIM>>>(matrix_d, b_d, b_next_d, rows, cols);
        
        
        // Reduction Kernel -> sum squared 
        squaredSumGPU(b_next_d, N, normSq_d, scratchA_d, scratchB_d);

        // fused: Element-wise norm + Element-wise abs_diff

        // Reduction Kernel -> sum abs_diff
        normalizeDiffSumGPU(b_next_d, b_d, normSq_d, diffSum_d,
            scratchA_d, scratchB_d, N);

        std::swap(b_d, b_next_d);
        // Compare diff & repeat
        float h_diff;
        cudaMemcpy(&h_diff, diffSum_d, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < tol) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy data back to the CPU
    cudaMemcpy(b_next, b_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU data
    cudaFree(matrix_d);
    cudaFree(b_d);
    cudaFree(b_next_d);
    cudaFree(normSq_d);
    cudaFree(diffSum_d);
    cudaFree(scratchA_d);
    cudaFree(scratchB_d);


    // Compute eigenvalue on CPU
    

    // Compute eigenvalue
    float eigenvalue = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            eigenvalue += b_next[i] * matrix[i*N + j] * b_next[j];
        }
    }
    *eigen_value = eigenvalue;
    

}


// Function to perform power iteration to find the largest eigenvalue and eigenvector
void powerIteration_cpu(float* matrix, float* b, float* b_next, int N, float* eigen_value, int maxIter = 100000, float tol = 1e-6) {

    for (int iter = 0; iter < maxIter; ++iter) {
        memset(b_next, 0, N * sizeof(float));

        // Multiply matrix with b
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                b_next[i] += matrix[i * N + j] * b[j];
            }
        }

        // Normalize b_next
        float norm = 0.0f;
        for (int i = 0; i < N; ++i) {
            norm += b_next[i] * b_next[i];
        }
        norm = sqrtf(norm);
        for (int i = 0; i < N; ++i) {
            b_next[i] /= norm;
        }

        // Check for convergence
        float diff = 0.0f;
        for (int i = 0; i < N; ++i) {
            diff += fabsf(b_next[i] - b[i]);
        }
        if (diff < tol) {
            break;
        }

        // Copy b_next into b for the next iteration
        memcpy(b, b_next, N * sizeof(float));
    }

    // Compute eigenvalue: λ = b^T * A * b
    *eigen_value = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            *eigen_value += b_next[i] * matrix[i * N + j] * b_next[j];
        }
    }
    
}

int main(int argc, char* argv[]) {
    
    // ######################### Allocate memory & initialize data #########################

    // Declare the host arrays
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1024);
    float *matrix = (float*)malloc(N * N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    float *b_next = (float*)malloc(N * sizeof(float));
    float eigen_value = 0;

    
    
    // Generate Random Numbers
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_int_distribution<int> dist(0, 255);

    // Initialize the arrays
    for (int i = 0; i < N*N; i++) {
        matrix[i] = dist(generator);
    }
    for(int i = 0; i < N; i++) {
        b[i] = 1.0f;
        b_next[i] = 0.0f;
    }
    // printArr(A, N * N);
    // printArr(B, N * N);

    // ######################### Power Iteration on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); 
    powerIteration_cpu(matrix, b, b_next, N, &eigen_value);
    end_cpu = high_resolution_clock::now(); 
    //////////////////////////////////////

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    std::cout << "Total CPU time: " << duration_sec.count() << "ms\n";
    std::cout << "Eigenvalue: " << eigen_value << std::endl;

    // ######################### vector addition on GPU #########################
    // Re-initialize host inputs (CPU run mutated b and b_next)
    for (unsigned int i = 0; i < N; ++i) { b[i] = 1.0f; b_next[i] = 0.0f; }
    eigen_value = 0.0f;

    
    
    // cudaEvent_t start_gpu;
    // cudaEvent_t stop_gpu;
    // cudaEventCreate(&start_gpu);
    // cudaEventCreate(&stop_gpu);

    // //////////////////////////////////////
    // cudaEventRecord(start_gpu);
    // matmul_gpu(A, B, C, N);
    // cudaEventRecord(stop_gpu);
    // cudaEventSynchronize(stop_gpu);
    // //////////////////////////////////////

    // // Get the elapsed time in milliseconds
    // float ms;
    // cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    // std::cout << "Total GPU time: " << ms << "ms\n";
    std::cout << "Eigenvalue: " << *eigen_value << std::endl;

    // free host memory
    free(matrix);
    free(b);

    return 0;
}