#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void compute_mean(const float *vec, float *mean, int size) {
    __shared__ float partial_sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load elements into shared memory
    partial_sum[tid] = (idx < size) ? vec[idx] : 0.0f;
    __syncthreads();

    // Perform reduction to calculate the sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // Store the block's sum in the output mean array
    if (tid == 0) {
        atomicAdd(mean, partial_sum[0]);
    }
}

__global__ void compute_stddev(const float *vec, float mean, float *stddev, int size) {
    __shared__ float partial_sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Calculate the squared difference from the mean
    float diff = (idx < size) ? vec[idx] - mean : 0.0f;
    partial_sum[tid] = diff * diff;
    __syncthreads();

    // Perform reduction to calculate the sum of squared differences
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // Store the block's sum in the output stddev array
    if (tid == 0) {
        atomicAdd(stddev, partial_sum[0]);
    }
}

__global__ void standardize_vector(float *vec, float mean, float stddev, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = (vec[idx] - mean) / stddev;
    }
}

void standardize(float *h_vec, int size) {
    float *d_vec, *d_mean, *d_stddev;
    float h_mean = 0.0f, h_stddev = 0.0f;

    cudaMalloc(&d_vec, size * sizeof(float));
    cudaMalloc(&d_mean, sizeof(float));
    cudaMalloc(&d_stddev, sizeof(float));

    cudaMemcpy(d_vec, h_vec, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, &h_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stddev, &h_stddev, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Compute mean
    compute_mean<<<numBlocks, blockSize>>>(d_vec, d_mean, size);
    cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    h_mean /= size;

    // Compute standard deviation
    compute_stddev<<<numBlocks, blockSize>>>(d_vec, h_mean, d_stddev, size);
    cudaMemcpy(&h_stddev, d_stddev, sizeof(float), cudaMemcpyDeviceToHost);
    h_stddev = sqrt(h_stddev / size);

    // Standardize the vector
    standardize_vector<<<numBlocks, blockSize>>>(d_vec, h_mean, h_stddev, size);

    // Copy the result back to host
    cudaMemcpy(h_vec, d_vec, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_mean);
    cudaFree(d_stddev);
}

int main() {
    int size = 1000;
    float *h_vec = (float *)malloc(size * sizeof(float));

    // Initialize the vector
    for (int i = 0; i < size; i++) {
        h_vec[i] = rand() % 100;
    }

    standardize(h_vec, size);

    // Print some results
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_vec[i]);
    }

    free(h_vec);
    return 0;
}
