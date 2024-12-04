#ifndef EIGEN_DECOMPOSITION_H
#define EIGEN_DECOMPOSITION_H

#include <vector>

// Function to compute eigenvalues and eigenvectors of a symmetric covariance matrix
void compute_eigenvalues_and_eigenvectors(const std::vector<std::vector<float>>& covariance_matrix,
                                           std::vector<float>& eigenvalues,
                                           std::vector<std::vector<float>>& eigenvectors);

// Function to create a feature vector based on principal components
std::vector<float> create_feature_vector(const std::vector<float>& image,
                                          const std::vector<std::vector<float>>& eigenvectors,
                                          int width, int height);

#endif // EIGEN_DECOMPOSITION_H
