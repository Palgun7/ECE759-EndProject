#include "standardize_matrix.h"
#include <vector>
#include <cmath>

// Standardize matrix to have mean 0 and variance 1 for each column
std::vector<std::vector<double>> standardize_matrix(const std::vector<std::vector<double>>& X) {
    size_t rows = X.size();
    size_t cols = X[0].size();

    std::vector<std::vector<double>> standardized_X(rows, std::vector<double>(cols));
    std::vector<double> means(cols, 0.0);
    std::vector<double> std_devs(cols, 0.0);

    // Calculate means
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            means[j] += X[i][j];
        }
        means[j] /= rows;
    }

    // Calculate standard deviations
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            std_devs[j] += (X[i][j] - means[j]) * (X[i][j] - means[j]);
        }
        std_devs[j] = std::sqrt(std_devs[j] / rows);
    }

    // Standardize matrix
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            standardized_X[i][j] = (X[i][j] - means[j]) / std_devs[j];
        }
    }

    return standardized_X;
}
