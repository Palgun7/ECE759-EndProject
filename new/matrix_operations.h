#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& matA, const std::vector<std::vector<double>>& matB);
std::vector<std::vector<double>> centerMatrix(const std::vector<std::vector<double>>& matrix);
std::vector<double> computeMean(const std::vector<std::vector<double>>& matrix);
// Normalizes a matrix (zero mean for each column)
std::vector<std::vector<double>> normalizeMatrix(const std::vector<std::vector<double>>& matrix);

// Computes the covariance matrix
std::vector<std::vector<double>> computeCovarianceMatrix(const std::vector<std::vector<double>>& normalizedMatrix);

#endif // MATRIX_OPERATIONS_H
