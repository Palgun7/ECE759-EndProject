#ifndef COVARIANCE_MATRIX_H
#define COVARIANCE_MATRIX_H

#include <vector>

// Function to compute the covariance matrix of a normalized grayscale image
std::vector<std::vector<float>> compute_covariance_matrix(const std::vector<float>& image, int width, int height);

#endif // COVARIANCE_MATRIX_H
