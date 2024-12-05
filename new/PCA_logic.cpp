#include "PCA_logic.h"
#include "matrix_operations.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>

const double EPSILON = 1e-6; // Convergence threshold for eigenvalue computation
const int MAX_ITERATIONS = 1000;

// Sorts eigenvalues and eigenvectors in descending order of eigenvalues
std::pair<std::vector<double>, std::vector<std::vector<double>>>
sortEigenvaluesAndVectors(const std::vector<double>& eigenvalues, const std::vector<std::vector<double>>& eigenvectors) {
    size_t n = eigenvalues.size();
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    // Sort indices based on eigenvalues
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return eigenvalues[a] > eigenvalues[b];
    });

    std::vector<double> sortedEigenvalues;
    std::vector<std::vector<double>> sortedEigenvectors;

    for (size_t i : indices) {
        sortedEigenvalues.push_back(eigenvalues[i]);
        sortedEigenvectors.push_back(eigenvectors[i]);
    }

    return {sortedEigenvalues, sortedEigenvectors};
}

// Selects top-k eigenvectors
std::vector<std::vector<double>> getTopKEigenvectors(const std::vector<std::vector<double>>& eigenvectors, int k) {
    return std::vector<std::vector<double>>(eigenvectors.begin(), eigenvectors.begin() + k);
}

// Projects data to principal components
std::vector<std::vector<double>> projectToPrincipalComponents(
    const std::vector<std::vector<double>>& normalizedMatrix,
    const std::vector<std::vector<double>>& principalComponents) {
    size_t rows = normalizedMatrix.size();
    size_t k = principalComponents.size();
    std::vector<std::vector<double>> projected(rows, std::vector<double>(k, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < principalComponents[j].size(); ++l) {
                projected[i][j] += normalizedMatrix[i][l] * principalComponents[j][l];
            }
        }
    }

    return projected;
}

// Helper function to create an identity matrix
std::vector<std::vector<double>> identityMatrix(size_t size) {
    std::vector<std::vector<double>> identity(size, std::vector<double>(size, 0.0));
    for (size_t i = 0; i < size; ++i) {
        identity[i][i] = 1.0;
    }
    return identity;
}

// Multiply a matrix by a scalar
std::vector<std::vector<double>> scalarMultiply(const std::vector<std::vector<double>>& matrix, double scalar) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}

// Power Iteration to compute the dominant eigenvalue and eigenvector
std::pair<double, std::vector<double>> powerIteration(const std::vector<std::vector<double>>& matrix) {
    size_t n = matrix.size();
    std::vector<double> eigenvector(n, 1.0); // Start with a random non-zero vector
    double eigenvalue = 0.0;

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        // Multiply matrix with eigenvector
        std::vector<double> nextVector(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                nextVector[i] += matrix[i][j] * eigenvector[j];
            }
        }

        // Normalize the resulting vector
        double norm = 0.0;
        for (double val : nextVector) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (size_t i = 0; i < n; ++i) {
            nextVector[i] /= norm;
        }

        // Check for convergence
        double nextEigenvalue = 0.0;
        for (size_t i = 0; i < n; ++i) {
            nextEigenvalue += nextVector[i] * (matrix[i][0] * nextVector[0]); // Rayleigh quotient approximation
        }

        if (std::fabs(nextEigenvalue - eigenvalue) < EPSILON) {
            eigenvalue = nextEigenvalue;
            eigenvector = nextVector;
            break;
        }

        eigenvalue = nextEigenvalue;
        eigenvector = nextVector;
    }

    return {eigenvalue, eigenvector};
}

// Solve (A - lambda * I)x = 0 for the eigenvector
std::vector<double> solveForEigenvector(const std::vector<std::vector<double>>& matrix, double eigenvalue) {
    size_t n = matrix.size();
    std::vector<std::vector<double>> adjustedMatrix = matrix;

    // Subtract lambda * I from A
    for (size_t i = 0; i < n; ++i) {
        adjustedMatrix[i][i] -= eigenvalue;
    }

    // Apply Gaussian elimination (basic implementation)
    std::vector<double> eigenvector(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (adjustedMatrix[i][i] == 0.0) {
            throw std::runtime_error("Matrix is singular or near singular.");
        }
        double pivot = adjustedMatrix[i][i];
        for (size_t j = i; j < n; ++j) {
            adjustedMatrix[i][j] /= pivot;
        }
        eigenvector[i] = 1.0; // Assume a non-zero solution
    }

    return eigenvector;
}

// Compute eigenvalues and eigenvectors for the covariance matrix
std::pair<std::vector<double>, std::vector<std::vector<double>>> computeEigen(const std::vector<std::vector<double>>& matrix) {
    size_t n = matrix.size();
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors;

    // Iteratively compute eigenvalues and eigenvectors
    std::vector<std::vector<double>> remainingMatrix = matrix;
    for (size_t i = 0; i < n; ++i) {
        auto [eigenvalue, eigenvector] = powerIteration(remainingMatrix);
        eigenvalues.push_back(eigenvalue);
        eigenvectors.push_back(eigenvector);

        // Deflate the matrix to find the next eigenvalue/eigenvector
        std::vector<std::vector<double>> rank1Update = scalarMultiply(
            multiply({eigenvector}, {eigenvector}), eigenvalue);
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < n; ++c) {
                remainingMatrix[r][c] -= rank1Update[r][c];
            }
        }
    }

    return {eigenvalues, eigenvectors};
}
