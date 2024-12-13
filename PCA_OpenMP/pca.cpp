#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include "pca.h"



using namespace std;

// // Function to compute the mean of each column
// vector<float> computeColumnMeans(const vector<vector<float>> &matrix)
// {
//     int rows = matrix.size();
//     int cols = matrix[0].size();
//     vector<float> means(cols, 0.0f);

//     for (int j = 0; j < cols; ++j)
//     {
//         for (int i = 0; i < rows; ++i)
//         {
//             means[j] += matrix[i][j];
//         }
//         means[j] /= rows;
//     }
//     return means;
// }

// vector<vector<float>> centerMatrix(const vector<vector<float>> &matrix, const vector<float> &means)
// {
//     int rows = matrix.size();
//     int cols = matrix[0].size();
//     vector<vector<float>> centered(rows, vector<float>(cols));

//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             centered[i][j] = matrix[i][j] - means[j];
//         }
//     }
//     return centered;
// }

// // Function to compute the transpose of a matrix
// vector<vector<float>> transpose(const vector<vector<float>> &matrix)
// {
//     int rows = matrix.size();
//     int cols = matrix[0].size();
//     vector<vector<float>> transposed(cols, vector<float>(rows));

//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             transposed[j][i] = matrix[i][j];
//         }
//     }
//     return transposed;
// }

// vector<vector<float>> multiply(const vector<vector<float>> &A, const vector<vector<float>> &B)
// {
//     int rows = A.size();
//     int cols = B[0].size();
//     int inner = A[0].size();

//     vector<vector<float>> result(rows, vector<float>(cols, 0.0f));

//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             for (int k = 0; k < inner; ++k)
//             {
//                 result[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
//     return result;
// }


// pair<float, vector<float>> powerIteration(const vector<vector<float>> &matrix, int maxIter = 100000, float tol = 1e-6)
// {
//     int n = matrix.size();
//     vector<float> b(n, 1.0f); // Initial vector
//     float eigenvalue = 0.0f;

//     for (int iter = 0; iter < maxIter; ++iter)
//     {
//         // if (iter % 1000 == 0)
//             // cout << "|";
//         vector<float> b_next(n, 0.0f);

//         // Multiply matrix with b
//         for (int i = 0; i < n; ++i)
//         {
//             for (int j = 0; j < n; ++j)
//             {
//                 b_next[i] += matrix[i][j] * b[j];
//             }
//         }

//         // Normalize b_next
//         float norm = 0.0f;
//         for (float val : b_next)
//         {
//             norm += val * val;
//         }
//         norm = sqrt(norm);
//         for (float &val : b_next)
//         {
//             val /= norm;
//         }

//         // Check for convergence
//         float diff = 0.0f;
//         for (int i = 0; i < n; ++i)
//         {
//             diff += abs(b_next[i] - b[i]);
//         }
//         if (diff < tol)
//         {
//             // Compute eigenvalue
//             eigenvalue = 0.0f;
//             for (int i = 0; i < n; ++i)
//             {
//                 for (int j = 0; j < n; ++j)
//                 {
//                     eigenvalue += b_next[i] * matrix[i][j] * b_next[j];
//                 }
//             }
//             return {eigenvalue, b_next};
//         }
//         b = b_next;
//     }

//     cerr << "Power iteration did not converge!" << endl;
//     return {eigenvalue, b};
// }

// // Comparator struct to sort indices based on eigenvalues
// struct CompareEigenvalues
// {
//     const std::vector<float> &eigenvalues;
//     CompareEigenvalues(const std::vector<float> &eigenvalues) : eigenvalues(eigenvalues) {}
//     bool operator()(int a, int b) const
//     {
//         return eigenvalues[a] > eigenvalues[b];
//     }
// };

// // Function to select the top N eigenvectors based on their eigenvalues
// std::vector<std::vector<float>> selectTopEigenvectors(const std::vector<std::vector<float>> &eigenvectors, const std::vector<float> &eigenvalues, int numComponents)
// {
//     // Debug output
//     // std::cout << "Number of eigenvectors: " << eigenvectors.size() << std::endl;
//     // std::cout << "Number of eigenvalues: " << eigenvalues.size() << std::endl;
//     // std::cout << "Number of components requested: " << numComponents << std::endl;

//     // Ensure numComponents is within the valid range
//     if (numComponents > eigenvectors.size() || numComponents > eigenvalues.size())
//     {
//         throw std::invalid_argument("numComponents is larger than the number of available eigenvectors or eigenvalues.");
//     }

//     // Create a vector of indices
//     std::vector<int> indices(eigenvalues.size());
//     std::iota(indices.begin(), indices.end(), 0);

//     // Sort indices based on eigenvalues in descending order using the comparator struct
//     std::sort(indices.begin(), indices.end(), CompareEigenvalues(eigenvalues));

//     // Debug output
//     // std::cout << "Indices sorted based on eigenvalues: ";
//     // for (int index : indices)
//     // {
//     //     std::cout << index << " ";
//     // }
//     // std::cout << std::endl;

//     // Select the top numComponents eigenvectors
//     std::vector<std::vector<float>> topEigenvectors(numComponents);
//     for (int i = 0; i < numComponents; ++i)
//     {
//         topEigenvectors[i] = eigenvectors[indices[i]];
//     }

//     return topEigenvectors;
// }

// // Function to project data onto the principal components
// vector<vector<float>> projectOntoPrincipalComponents(const vector<vector<float>> &centered, const vector<vector<float>> &eigenvectors)
// {
//     int rows = centered.size();
//     int numComponents = eigenvectors.size();
//     vector<vector<float>> projected(rows, vector<float>(numComponents, 0.0f));

//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < numComponents; ++j)
//         {
//             for (int k = 0; k < centered[0].size(); ++k)
//             {
//                 projected[i][j] += centered[i][k] * eigenvectors[j][k];
//             }
//         }
//     }
//     return projected;
// }

// // Function to reconstruct data from the principal components
// std::vector<std::vector<float>> reconstructFromPrincipalComponents(const std::vector<std::vector<float>> &projected, const std::vector<std::vector<float>> &eigenvectors, const std::vector<float> &means)
// {
//     int rows = projected.size();
//     int cols = eigenvectors[0].size();
//     int numComponents = eigenvectors.size();

//     std::vector<std::vector<float>> reconstructed(rows, std::vector<float>(cols, 0.0f));

//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             for (int k = 0; k < numComponents; ++k)
//             {
//                 reconstructed[i][j] += projected[i][k] * eigenvectors[k][j];
//             }
//             reconstructed[i][j] += means[j];
//         }
//     }

//     return reconstructed;
// }


// Function to compute the transpose of a matrix
vector<vector<float>> transpose(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<float>> transposed(cols, vector<float>(rows));

    #pragma omp for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // #pragma omp atomic
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Function to multiply two matrices
vector<vector<float>> multiply(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    int rows = A.size();
    int cols = B[0].size();
    int inner = A[0].size();

    vector<vector<float>> result(rows, vector<float>(cols, 0.0f));

    #pragma omp for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < inner; ++k) {
                #pragma omp atomic
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Function to compute the mean of each column
vector<float> computeColumnMeans(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<float> means(cols, 0.0f);

    #pragma omp for collapse (1)
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            #pragma omp atomic
            means[j] += matrix[i][j];
        }
        means[j] /= rows;
    }
    return means;
}

// Function to center the matrix
vector<vector<float>> centerMatrix(const vector<vector<float>>& matrix, const vector<float>& means) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<float>> centered(rows, vector<float>(cols));

    #pragma omp for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // #pragma omp atomic
            centered[i][j] = matrix[i][j] - means[j];
        }
    }
    return centered;
}

// Function to perform power iteration to find the largest eigenvalue and eigenvector
pair<float, vector<float>> powerIteration(const vector<vector<float>>& matrix, int maxIter = 100000, float tol = 1e-6) {
    int n = matrix.size();
    vector<float> b(n, 1.0f); // Initial vector
    float eigenvalue = 0.0f;

    for (int iter = 0; iter < maxIter; ++iter) {
        if(iter % 1000 == 0) 
            cout << "|";
        vector<float> b_next(n, 0.0f);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                b_next[i] += matrix[i][j] * b[j];
            }
        }

        // Normalize b_next
        float norm = 0.0f;
        #pragma omp parallel for reduction(+:norm)
        for (float val : b_next) {
            norm += val * val;
        }
        norm = sqrt(norm);
        #pragma omp parallel for
        for (float& val : b_next) {
            val /= norm;
        }

        // Check for convergence
        float diff = 0.0f;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < n; ++i) {
            diff += abs(b_next[i] - b[i]);
        }
        if (diff < tol) {
            // Compute eigenvalue
            eigenvalue = 0.0f;
            #pragma omp parallel for reduction(+:eigenvalue)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
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
struct CompareEigenvalues {
    const std::vector<float>& eigenvalues;
    CompareEigenvalues(const std::vector<float>& eigenvalues) : eigenvalues(eigenvalues) {}
    bool operator()(int a, int b) const {
        return eigenvalues[a] > eigenvalues[b];
    }
};

// Function to select the top N eigenvectors based on their eigenvalues
std::vector<std::vector<float>> selectTopEigenvectors(const std::vector<std::vector<float>>& eigenvectors, const std::vector<float>& eigenvalues, int numComponents) {
    // Ensure numComponents is within the valid range
    if (numComponents > eigenvectors.size() || numComponents > eigenvalues.size()) {
        throw std::invalid_argument("numComponents is larger than the number of available eigenvectors or eigenvalues.");
    }

    // Create a vector of indices
    std::vector<int> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on eigenvalues in descending order using the comparator struct
    std::sort(indices.begin(), indices.end(), CompareEigenvalues(eigenvalues));

    // Select the top numComponents eigenvectors
    std::vector<std::vector<float>> topEigenvectors(numComponents);
    #pragma omp parallel for
    for (int i = 0; i < numComponents; ++i) {
        topEigenvectors[i] = eigenvectors[indices[i]];
    }

    return topEigenvectors;
}

// Function to project data onto the principal components
vector<vector<float>> projectOntoPrincipalComponents(const vector<vector<float>>& centered, const vector<vector<float>>& eigenvectors) {
    int rows = centered.size();
    int numComponents = eigenvectors.size();
    vector<vector<float>> projected(rows, vector<float>(numComponents, 0.0f));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < numComponents; ++j) {
            for (int k = 0; k < centered[0].size(); ++k) {
                projected[i][j] += centered[i][k] * eigenvectors[j][k];
            }
        }
    }
    return projected;
}

// Function to reconstruct data from the principal components
std::vector<std::vector<float>> reconstructFromPrincipalComponents(const std::vector<std::vector<float>>& projected, const std::vector<std::vector<float>>& eigenvectors, const std::vector<float>& means) {
    int rows = projected.size();
    int cols = eigenvectors[0].size();
    int numComponents = eigenvectors.size();
    
    std::vector<std::vector<float>> reconstructed(rows, std::vector<float>(cols, 0.0f));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < numComponents; ++k) {
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
    auto means = computeColumnMeans(imageMatrix);
    auto centered = centerMatrix(imageMatrix, means);

    // Compute covariance matrix
    auto transposed = transpose(centered);
    auto covariance = multiply(transposed, centered);

    // cout << "\nCovariance matrix calculated\n";

    // Perform power iteration for the largest eigenvalue and eigenvector
    // auto [eigenvalue, eigenvector] = powerIteration(covariance);
    // // std::cout << "Power Iteration Complete \n";

    // // Convert eigenvector to matrix form
    // vector<vector<float>> eigenvectors = {eigenvector};
    // vector<float> eigenvalues = {eigenvalue};

    // // cout << "Eigen Values : ";
    // // for (int i; i < eigenvalues.size(); i++)
    // // {
    // //     cout << eigenvalues[i];
    // // }
    // // cout << "\n";
    // int numComponents = eigenvalues.size();

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