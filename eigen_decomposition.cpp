#include "eigen_decomposition.h"
#include <cmath>
#include <algorithm>

constexpr double EPSILON = 1e-9;

// Helper function to compute the norm of a vector
double vector_norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double val : v) sum += val * val;
    return std::sqrt(sum);
}

// Helper function for matrix-vector multiplication
std::vector<double> multiply_matrix_vector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
    size_t size = matrix.size();
    std::vector<double> result(size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

// Power iteration to compute the dominant eigenvalue and eigenvector
std::pair<double, std::vector<double>> power_iteration(const std::vector<std::vector<double>>& matrix, int max_iter = 1000) {
    size_t size = matrix.size();
    std::vector<double> b(size, 1.0);  // Initial guess for eigenvector
    std::vector<double> b_next(size, 0.0);  // To hold the updated eigenvector

    for (int iter = 0; iter < max_iter; ++iter) {
        // Multiply matrix with vector
        b_next = multiply_matrix_vector(matrix, b);

        // Normalize the vector
        double norm = vector_norm(b_next);
        for (size_t i = 0; i < size; ++i) {
            b_next[i] /= norm;
        }

        // Check for convergence
        if (vector_norm(b_next) - vector_norm(b) < EPSILON) break;

        b = b_next;
    }

    // Calculate the eigenvalue using the Rayleigh quotient
    double eigenvalue = 0.0;
    std::vector<double> Ax = multiply_matrix_vector(matrix, b);
    for (size_t i = 0; i < size; ++i) {
        eigenvalue += b[i] * Ax[i];
    }

    return {eigenvalue, b};
}


// Full Eigenvalue Decomposition using Power Iteration
std::pair<std::vector<double>, std::vector<std::vector<double>>> eigen_decomposition(const std::vector<std::vector<double>>& covariance_matrix) {
    size_t size = covariance_matrix.size();
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors;

    std::vector<std::vector<double>> matrix = covariance_matrix;
    for (size_t k = 0; k < size; ++k) {
        auto [eigenvalue, eigenvector] = power_iteration(matrix);
        eigenvalues.push_back(eigenvalue);
        eigenvectors.push_back(eigenvector);

        // Deflate the matrix: remove the contribution of the current eigenvector
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    return {eigenvalues, eigenvectors};
}
