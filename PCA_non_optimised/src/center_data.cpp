#include "center_data.h"
#include <vector>
#include <numeric>
#include <iostream>

std::vector<std::vector<float>> centerData(const std::vector<std::vector<float>>& imageMatrix) {
    if (imageMatrix.empty() || imageMatrix[0].empty()) {
        std::cerr << "Error: The image matrix is empty." << std::endl;
        return {};
    }

    int numRows = imageMatrix.size();    // Number of rows (M)
    int numCols = imageMatrix[0].size(); // Number of columns (N^2)

    // Step 1: Calculate the column-wise mean
    std::vector<float> columnMeans(numCols, 0.0f);
    for (int col = 0; col < numCols; ++col) {
        for (int row = 0; row < numRows; ++row) {
            columnMeans[col] += imageMatrix[row][col];
        }
        columnMeans[col] /= numRows;
    }

    // Step 2: Subtract the mean from each element in the matrix
    std::vector<std::vector<float>> centeredMatrix(numRows, std::vector<float>(numCols, 0.0f));
    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numCols; ++col) {
            centeredMatrix[row][col] = imageMatrix[row][col] - columnMeans[col];
        }
    }

    return centeredMatrix;
}
