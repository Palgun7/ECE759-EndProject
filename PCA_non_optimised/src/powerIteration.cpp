#include "powerIteration.h"
#include <cmath>
#include <iostream>

// Function to multiply matrix with vector
std::vector<float> multiply(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    std::vector<float> result(vec.size(), 0.0f);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

// Function to normalize a vector
void normalize(std::vector<float>& vec) {
    float norm = 0.0f;
    for (size_t i = 0; i < vec.size(); ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] /= norm;
    }
}

// Power iteration function
std::vector<float> powerIterationSingle(const std::vector<std::vector<float>>& matrix, int maxIterations, float tolerance) {
    std::vector<float> b_k(matrix.size(), 1.0f); // Initial vector
    normalize(b_k);

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<float> b_k1 = multiply(matrix, b_k);
        normalize(b_k1);

        // Check for convergence
        float diff = 0.0f;
        for (size_t i = 0; i < b_k.size(); ++i) {
            diff += std::abs(b_k1[i] - b_k[i]);
        }

        if (diff < tolerance) {
            break;
        }

        b_k = b_k1;
    }

    return b_k;
}

// Function to deflate the matrix
void deflateMatrix(std::vector<std::vector<float>>& matrix, const std::vector<float>& eigenvector, float eigenvalue) {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
}

// Main power iteration function to compute all eigenvectors
std::vector<std::vector<float>> powerIteration(const std::vector<std::vector<float>>& CovarianceMatrix, int maxIterations, float tolerance) {
    std::vector<std::vector<float>> matrix = CovarianceMatrix; // Copy of the input matrix
    std::vector<std::vector<float>> eigenvectors;
    int n = matrix.size();

    for (int i = 0; i < n; ++i) {
        std::vector<float> eigenvector = powerIterationSingle(matrix, maxIterations, tolerance);
        float eigenvalue = 0.0f;
        for (size_t j = 0; j < matrix.size(); ++j) {
            eigenvalue += eigenvector[j] * multiply(matrix, eigenvector)[j];
        }
        eigenvectors.push_back(eigenvector);
        deflateMatrix(matrix, eigenvector, eigenvalue);
    }

    return eigenvectors;
}
