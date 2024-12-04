#include "covariance_matrix.h"
#include <vector>

std::vector<std::vector<float>> compute_covariance_matrix(const std::vector<float>& image, int width, int height) {
    int n_samples = height;  // Rows represent samples
    int n_features = width; // Columns represent features

    // Initialize covariance matrix
    std::vector<std::vector<float>> covariance_matrix(n_features, std::vector<float>(n_features, 0.0f));

    // Compute covariance matrix
    for (int i = 0; i < n_features; ++i) {
        for (int j = 0; j < n_features; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < n_samples; ++k) {
                float xi = image[k * n_features + i]; // Feature i for sample k
                float xj = image[k * n_features + j]; // Feature j for sample k
                sum += xi * xj;
            }

            covariance_matrix[i][j] = sum / (n_samples - 1);
        }
    }

    return covariance_matrix;
}
