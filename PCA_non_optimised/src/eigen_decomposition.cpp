#include "eigen_decomposition.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

// Power iteration to approximate the largest eigenvalue and its eigenvector
void power_iteration(const std::vector<std::vector<float>>& matrix,
                     std::vector<float>& eigenvector, float& eigenvalue, int max_iters = 1000, float tol = 1e-6) {
    int size = matrix.size();
    eigenvector.assign(size, 1.0f); // Initialize eigenvector with 1s
    std::vector<float> temp(size, 0.0f);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Multiply matrix by eigenvector: temp = matrix * eigenvector
        for (int i = 0; i < size; ++i) {
            temp[i] = 0.0f;
            for (int j = 0; j < size; ++j) {
                temp[i] += matrix[i][j] * eigenvector[j];
            }
        }

        // Normalize temp to create the new eigenvector
        float norm = std::sqrt(std::inner_product(temp.begin(), temp.end(), temp.begin(), 0.0f));
        for (int i = 0; i < size; ++i) {
            temp[i] /= norm;
        }

        // Check for convergence
        float diff = 0.0f;
        for (int i = 0; i < size; ++i) {
            diff += std::fabs(temp[i] - eigenvector[i]);
        }
        if (diff < tol) {
            break;
        }

        eigenvector = temp; // Update eigenvector
    }

    // Compute eigenvalue
    eigenvalue = 0.0f;
    for (int i = 0; i < size; ++i) {
        float projection = 0.0f;
        for (int j = 0; j < size; ++j) {
            projection += matrix[i][j] * eigenvector[j];
        }
        eigenvalue += projection * eigenvector[i];
    }
}

// Compute all eigenvalues and eigenvectors of a symmetric matrix
void compute_eigenvalues_and_eigenvectors(const std::vector<std::vector<float>>& covariance_matrix,
                                           std::vector<float>& eigenvalues,
                                           std::vector<std::vector<float>>& eigenvectors) {
    int size = covariance_matrix.size();
    eigenvalues.resize(size);
    eigenvectors.resize(size, std::vector<float>(size));

    // Perform iterative eigenvalue and eigenvector computation
    std::vector<std::vector<float>> matrix = covariance_matrix;
    for (int i = 0; i < size; ++i) {
        std::vector<float> eigenvector(size);
        float eigenvalue;
        power_iteration(matrix, eigenvector, eigenvalue);

        eigenvalues[i] = eigenvalue;
        eigenvectors[i] = eigenvector;

        // Deflate the matrix to find the next eigenvalue
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                matrix[j][k] -= eigenvalue * eigenvector[j] * eigenvector[k];
            }
        }
    }
}

// Create a feature vector based on the principal components
std::vector<float> create_feature_vector(const std::vector<float>& image,
                                          const std::vector<std::vector<float>>& eigenvectors,
                                          int width, int height) {
    int size = width * height;
    int num_features = eigenvectors.size();

    // Project the image data onto the top principal components
    std::vector<float> feature_vector(num_features);
    for (int i = 0; i < num_features; ++i) {
        feature_vector[i] = 0.0f;
        for (int j = 0; j < size; ++j) {
            feature_vector[i] += image[j] * eigenvectors[i][j];
        }
    }

    return feature_vector;
}
