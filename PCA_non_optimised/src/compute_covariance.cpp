#include "compute_covariance.h"
#include <vector>
#include <iostream>

std::vector<std::vector<float>> computeCovarianceMatrix(const std::vector<std::vector<float>>& centeredMatrix) {
    if (centeredMatrix.empty() || centeredMatrix[0].empty()) {
        std::cerr << "Error: The centered matrix is empty." << std::endl;
        return {};
    }

    int numRows = centeredMatrix.size();    // Number of rows (M)
    int numCols = centeredMatrix[0].size(); // Number of columns (N^2)

    // Initialize the covariance matrix with zeros
    std::vector<std::vector<float>> covarianceMatrix(numCols, std::vector<float>(numCols, 0.0f));

    // Compute the covariance matrix
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numCols; ++j) {
            float covariance = 0.0f;
            for (int k = 0; k < numRows; ++k) {
                covariance += centeredMatrix[k][i] * centeredMatrix[k][j];
            }
            covarianceMatrix[i][j] = covariance / (numRows - 1); // Sample covariance
        }
    }

    return covarianceMatrix;
}
