#include "covariance_matrix.h"

std::vector<std::vector<double>> compute_covariance_matrix(const std::vector<std::vector<double>>& X) {
    size_t rows = X.size();
    size_t cols = X[0].size();
    std::vector<std::vector<double>> covariance_matrix(cols, std::vector<double>(cols, 0.0));

    // Compute covariance matrix
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < rows; ++k) {
                covariance_matrix[i][j] += X[k][i] * X[k][j];
            }
            covariance_matrix[i][j] /= (rows);
        }
    }

    return covariance_matrix;
}

