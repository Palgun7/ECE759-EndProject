#include "image_reader.h"
#include "matrix_operations.h"
#include "PCA_logic.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    // Step 1: Read the PNG image
    const char* file_path = "data/image.png";
    int width = 0, height = 0;

    std::vector<float> image = read_png_file(file_path, width, height);

    if (width != 32 || height != 32) {
        std::cerr << "Error: Input image must be 32x32 pixels." << std::endl;
        return -1;
    }

    // Step 2: Convert the image to a 32x32 matrix
    std::vector<std::vector<double>> matrix(height, std::vector<double>(width, 0.0));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i][j] = static_cast<double>(image[i * width + j]);
        }
    }

    // Step 3: Normalize the matrix (zero mean for each column)
    std::vector<std::vector<double>> normalizedMatrix = normalizeMatrix(matrix);

    // Step 4: Compute the covariance matrix
    std::vector<std::vector<double>> covarianceMatrix = computeCovarianceMatrix(normalizedMatrix);

    // Step 5: Compute eigenvalues and eigenvectors of the covariance matrix
    auto [eigenvalues, eigenvectors] = computeEigen(covarianceMatrix);

    // Step 6: Sort eigenvalues and eigenvectors
    auto [sortedEigenvalues, sortedEigenvectors] = sortEigenvaluesAndVectors(eigenvalues, eigenvectors);

    // Step 7: Select top k eigenvectors (assume k=2 for 2D PCA)
    int k = 2;
    std::vector<std::vector<double>> principalComponents = getTopKEigenvectors(sortedEigenvectors, k);

    // Step 8: Transform the data into the reduced k-dimensional space
    std::vector<std::vector<double>> reducedData = projectToPrincipalComponents(normalizedMatrix, principalComponents);

    // Output the reduced data
    std::cout << "Reduced Data (2D Projection):" << std::endl;
    for (const auto& row : reducedData) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
