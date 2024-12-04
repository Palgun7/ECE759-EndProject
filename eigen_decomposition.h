#ifndef EIGEN_DECOMPOSITION_H
#define EIGEN_DECOMPOSITION_H

#include <vector>
#include <utility>

std::pair<std::vector<double>, std::vector<std::vector<double>>> eigen_decomposition(const std::vector<std::vector<double>>& covariance_matrix);

// Declare multiply_matrix_vector to be accessible in main.cpp
std::vector<double> multiply_matrix_vector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec);

#endif // EIGEN_DECOMPOSITION_H
