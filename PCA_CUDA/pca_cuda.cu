#include <cuda.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

using namespace std;

// CUDA kernel for matrix transpose
__global__ void transposeKernel(const float* input, float* output, int rows, int cols) {
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform the transpose if within bounds
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Host function to call the kernel
void transposeCUDA(const std::vector<std::vector<float>>& matrix, std::vector<std::vector<float>>& transposed) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Flatten the 2D matrix for CUDA processing
    std::vector<float> flatInput(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatInput[i * cols + j] = matrix[i][j];
        }
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // Copy input matrix to device memory
    cudaMemcpy(d_input, flatInput.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // Adjust based on hardware and problem size
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    transposeKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);

    // Allocate host memory for the transposed matrix
    std::vector<float> flatTransposed(rows * cols);

    // Copy the transposed matrix from device memory to host
    cudaMemcpy(flatTransposed.data(), d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert the flat output back to a 2D matrix
    transposed.resize(cols, std::vector<float>(rows));
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            transposed[i][j] = flatTransposed[i * rows + j];
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    if (row < rowsA && col < colsB) {
        float value = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            value += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = value;
    }
}

// Host function to perform matrix multiplication using CUDA
void matrixMultiplyCUDA(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B, std::vector<std::vector<float>>& C) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    // Flatten input matrices for CUDA processing
    std::vector<float> flatA(rowsA * colsA);
    std::vector<float> flatB(rowsB * colsB);
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsA; ++j)
            flatA[i * colsA + j] = A[i][j];
    for (int i = 0; i < rowsB; ++i)
        for (int j = 0; j < colsB; ++j)
            flatB[i * colsB + j] = B[i][j];

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rowsA * colsA * sizeof(float));
    cudaMalloc(&d_B, rowsB * colsB * sizeof(float));
    cudaMalloc(&d_C, rowsA * colsB * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, flatA.data(), rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // Adjust based on hardware
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Allocate host memory for the result
    std::vector<float> flatC(rowsA * colsB);

    // Copy the result from device to host
    cudaMemcpy(flatC.data(), d_C, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Reshape the result into a 2D matrix
    C.resize(rowsA, std::vector<float>(colsB));
    for (int i = 0; i < rowsA; ++i)
        for (int j = 0; j < colsB; ++j)
            C[i][j] = flatC[i * colsB + j];

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


// CUDA kernel for column-wise sum
__global__ void computeColumnSumsKernel(const float* matrix, float* columnSums, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += matrix[row * cols + col];
        }
        columnSums[col] = sum;
    }
}

// Host function to compute column means
void computeColumnMeansCUDA(const std::vector<std::vector<float>>& matrix, std::vector<float>& columnMeans) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Flatten the input matrix
    std::vector<float> flatMatrix(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatMatrix[i * cols + j] = matrix[i][j];
        }
    }

    // Allocate device memory
    float* d_matrix;
    float* d_columnSums;
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMalloc(&d_columnSums, cols * sizeof(float));

    // Copy matrix data to device
    cudaMemcpy(d_matrix, flatMatrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockDim = 256;
    int gridDim = (cols + blockDim - 1) / blockDim;

    // Launch the kernel to compute column sums
    computeColumnSumsKernel<<<gridDim, blockDim>>>(d_matrix, d_columnSums, rows, cols);

    // Allocate memory for column sums on the host
    std::vector<float> columnSums(cols);

    // Copy column sums back to the host
    cudaMemcpy(columnSums.data(), d_columnSums, cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute column means
    columnMeans.resize(cols);
    for (int j = 0; j < cols; ++j) {
        columnMeans[j] = columnSums[j] / rows;
    }

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_columnSums);
}

__global__ void centerMatrixKernel(const float* matrix, const float* means, float* centered, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        centered[index] = matrix[index] - means[col];
    }
}

std::vector<std::vector<float>> centerMatrixCUDA(const std::vector<std::vector<float>>& matrix, const std::vector<float>& means) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Flatten the 2D matrix into 1D array
    std::vector<float> matrix_flat(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix_flat[i * cols + j] = matrix[i][j];
        }
    }

    // Allocate memory on the device
    float *d_matrix, *d_means, *d_centered;
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMalloc(&d_means, cols * sizeof(float));
    cudaMalloc(&d_centered, rows * cols * sizeof(float));

    // Copy data to the device
    cudaMemcpy(d_matrix, matrix_flat.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, means.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define the block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    centerMatrixKernel<<<gridSize, blockSize>>>(d_matrix, d_means, d_centered, rows, cols);

    // Copy the result back to the host
    std::vector<float> centered_flat(rows * cols);
    cudaMemcpy(centered_flat.data(), d_centered, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_means);
    cudaFree(d_centered);

    // Reshape the result back into a 2D vector
    std::vector<std::vector<float>> centered(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            centered[i][j] = centered_flat[i * cols + j];
        }
    }

    return centered;
}

pair<float, vector<float>> powerIteration(const vector<vector<float>> &matrix, int maxIter = 100000, float tol = 1e-6)
{
    int n = matrix.size();
    vector<float> b(n, 1.0f); // Initial vector
    float eigenvalue = 0.0f;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        // if (iter % 1000 == 0)
            // cout << "|";
        vector<float> b_next(n, 0.0f);

        // Multiply matrix with b
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                b_next[i] += matrix[i][j] * b[j];
            }
        }

        // Normalize b_next
        float norm = 0.0f;
        for (float val : b_next)
        {
            norm += val * val;
        }
        norm = sqrt(norm);
        for (float &val : b_next)
        {
            val /= norm;
        }

        // Check for convergence
        float diff = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            diff += abs(b_next[i] - b[i]);
        }
        if (diff < tol)
        {
            // Compute eigenvalue
            eigenvalue = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    eigenvalue += b_next[i] * matrix[i][j] * b_next[j];
                }
            }
            return {eigenvalue, b_next};
        }
        b = b_next;
    }

    cerr << "Power iteration did not converge!" << endl;
    return {eigenvalue, b};
}

// Comparator struct to sort indices based on eigenvalues
struct CompareEigenvalues
{
    const std::vector<float> &eigenvalues;
    CompareEigenvalues(const std::vector<float> &eigenvalues) : eigenvalues(eigenvalues) {}
    bool operator()(int a, int b) const
    {
        return eigenvalues[a] > eigenvalues[b];
    }
};

// Function to select the top N eigenvectors based on their eigenvalues
std::vector<std::vector<float>> selectTopEigenvectors(const std::vector<std::vector<float>> &eigenvectors, const std::vector<float> &eigenvalues, int numComponents)
{
    // Debug output
    // std::cout << "Number of eigenvectors: " << eigenvectors.size() << std::endl;
    // std::cout << "Number of eigenvalues: " << eigenvalues.size() << std::endl;
    // std::cout << "Number of components requested: " << numComponents << std::endl;

    // Ensure numComponents is within the valid range
    if (numComponents > eigenvectors.size() || numComponents > eigenvalues.size())
    {
        throw std::invalid_argument("numComponents is larger than the number of available eigenvectors or eigenvalues.");
    }

    // Create a vector of indices
    std::vector<int> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on eigenvalues in descending order using the comparator struct
    std::sort(indices.begin(), indices.end(), CompareEigenvalues(eigenvalues));

    // Debug output
    // std::cout << "Indices sorted based on eigenvalues: ";
    // for (int index : indices)
    // {
    //     std::cout << index << " ";
    // }
    // std::cout << std::endl;

    // Select the top numComponents eigenvectors
    std::vector<std::vector<float>> topEigenvectors(numComponents);
    for (int i = 0; i < numComponents; ++i)
    {
        topEigenvectors[i] = eigenvectors[indices[i]];
    }

    return topEigenvectors;
}

// Function to project data onto the principal components
vector<vector<float>> projectOntoPrincipalComponents(const vector<vector<float>> &centered, const vector<vector<float>> &eigenvectors)
{
    int rows = centered.size();
    int numComponents = eigenvectors.size();
    vector<vector<float>> projected(rows, vector<float>(numComponents, 0.0f));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < numComponents; ++j)
        {
            for (int k = 0; k < centered[0].size(); ++k)
            {
                projected[i][j] += centered[i][k] * eigenvectors[j][k];
            }
        }
    }
    return projected;
}

// Function to reconstruct data from the principal components
std::vector<std::vector<float>> reconstructFromPrincipalComponents(const std::vector<std::vector<float>> &projected, const std::vector<std::vector<float>> &eigenvectors, const std::vector<float> &means)
{
    int rows = projected.size();
    int cols = eigenvectors[0].size();
    int numComponents = eigenvectors.size();

    std::vector<std::vector<float>> reconstructed(rows, std::vector<float>(cols, 0.0f));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            for (int k = 0; k < numComponents; ++k)
            {
                reconstructed[i][j] += projected[i][k] * eigenvectors[k][j];
            }
            reconstructed[i][j] += means[j];
        }
    }

    return reconstructed;
}

std::vector<float> pca(vector<vector<float>> imageMatrix, int blockSize, int imageWidth, int imageHeight)
{
    // Ensure the matrix is suitable for PCA
    int rows = imageMatrix.size();
    int cols = blockSize * blockSize;

    // Center the data
    std::vector<float> columnMeans;
    computeColumnMeansCUDA(imageMatrix, columnMeans);
    auto centered = centerMatrixCUDA(imageMatrix, columnMeans);    

    // Compute covariance matrix
    std::vector<std::vector<float>> transposed;
    transposeCUDA(centered, transposed);
    std::vector<std::vector<float>> covariance;
    matrixMultiplyCUDA(transposed, centered, covariance);

    // cout << "\nCovariance matrix calculated\n";

    // Perform power iteration for the largest eigenvalue and eigenvector
    auto [eigenvalue, eigenvector] = powerIteration(covariance);
    // std::cout << "Power Iteration Complete \n";

    // Convert eigenvector to matrix form
    vector<vector<float>> eigenvectors = {eigenvector};
    vector<float> eigenvalues = {eigenvalue};

    // cout << "Eigen Values : ";
    // for (int i; i < eigenvalues.size(); i++)
    // {
    //     cout << eigenvalues[i];
    // }
    // cout << "\n";
    int numComponents = eigenvalues.size();

    // std::vector<std::vector<float>> topEigenvectors = selectTopEigenvectors(eigenvectors, eigenvalues, numComponents);
    // cout << "Feature vector size : " << topEigenvectors[0].size() << " x" << topEigenvectors.size() << "\n";
    // auto projected = projectOntoPrincipalComponents(centered, topEigenvectors);
    // size_t numRows = projected.size();
    // size_t numCols = projected.empty() ? 0 : projected[0].size();
    // cout << "Projected vector size : " << numRows << " X " << numCols;

    // Reconstruct the blocks from the principal components
        // auto reconstructedBlocks = reconstructFromPrincipalComponents(projected, topEigenvectors, means);

        // Reassemble the blocks into the original image format
        vector<float> output_image_data(imageWidth * imageHeight, 0.0f);
        // int numBlocks = imageWidth / blockSize;
        // for (int rowBlock = 0; rowBlock < numBlocks; ++rowBlock)
        // {
        //     for (int colBlock = 0; colBlock < numBlocks; ++colBlock)
        //     {
        //         int blockIndex = rowBlock * numBlocks + colBlock;
        //         const auto &block = reconstructedBlocks[blockIndex];
        //         for (int i = 0; i < blockSize; ++i)
        //         {
        //             for (int j = 0; j < blockSize; ++j)
        //             {
        //                 int srcRow = rowBlock * blockSize + i;
        //                 int srcCol = colBlock * blockSize + j;
        //                 output_image_data[srcRow * imageWidth + srcCol] = block[i * blockSize + j];
        //             }
        //         }
        //     }
        // }
        return output_image_data;
}
