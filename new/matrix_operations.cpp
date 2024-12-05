#include "matrix_operations.h"
#include <vector>
#include <numeric>
#include <cmath>


// Normalizes a matrix (zero mean for each column)
std::vector<std::vector<double>> normalizeMatrix(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> normalized(rows, std::vector<double>(cols, 0.0));

    for (size_t j = 0; j < cols; ++j) {
        // Compute mean of column j
        double mean = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            mean += matrix[i][j];
        }
        mean /= rows;

        // Normalize column j
        for (size_t i = 0; i < rows; ++i) {
            normalized[i][j] = matrix[i][j] - mean;
        }
    }
    return normalized;
}

// Computes the covariance matrix
std::vector<std::vector<double>> computeCovarianceMatrix(const std::vector<std::vector<double>>& normalizedMatrix) {
    size_t rows = normalizedMatrix.size();
    size_t cols = normalizedMatrix[0].size();
    std::vector<std::vector<double>> covarianceMatrix(cols, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < rows; ++k) {
                sum += normalizedMatrix[k][i] * normalizedMatrix[k][j];
            }
            covarianceMatrix[i][j] = sum / (rows - 1);
        }
    }
    return covarianceMatrix;
}


// Compute the transpose of a matrix
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Multiply two matrices
std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& matA, const std::vector<std::vector<double>>& matB) {
    size_t rowsA = matA.size();
    size_t colsA = matA[0].size();
    size_t colsB = matB[0].size();
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    return result;
}

// Compute the mean of each column
std::vector<double> computeMean(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<double> mean(cols, 0.0);

    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            mean[j] += matrix[i][j];
        }
        mean[j] /= rows;
    }
    return mean;
}

// Center the matrix
std::vector<std::vector<double>> centerMatrix(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<double> mean = computeMean(matrix);
    std::vector<std::vector<double>> centered(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            centered[i][j] = matrix[i][j] - mean[j];
        }
    }
    return centered;
}
