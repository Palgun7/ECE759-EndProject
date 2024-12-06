#include <iostream>
<<<<<<< Updated upstream
#include <vector>
#include <cmath>

// Function to multiply two matrices
std::vector<std::vector<float>> matrixMultiply(const std::vector<std::vector<float>>& matA, const std::vector<std::vector<float>>& matB) {
    size_t rowsA = matA.size(), colsA = matA[0].size();
    size_t colsB = matB[0].size();
    std::vector<std::vector<float>> result(rowsA, std::vector<float>(colsB, 0.0f));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    return result;
}

// Function to normalize a vector
void normalizeVector(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    for (float& v : vec) {
        v /= norm;
    }
}

// Function to compute the dot product of two vectors
float dotProduct(const std::vector<float>& vecA, const std::vector<float>& vecB) {
    float result = 0.0f;
    for (size_t i = 0; i < vecA.size(); ++i) {
        result += vecA[i] * vecB[i];
    }
    return result;
}

// Function to compute the matrix-vector product
std::vector<float> matrixVectorMultiply(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec) {
    std::vector<float> result(mat.size(), 0.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[i].size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

// Power iteration to compute the largest eigenvalue and its eigenvector
std::pair<float, std::vector<float>> powerIteration(const std::vector<std::vector<float>>& matrix, int maxIterations = 1000, float tolerance = 1e-4) {
    size_t n = matrix.size();
    std::vector<float> eigenvector(n, 1.0f); // Start with an arbitrary vector
    normalizeVector(eigenvector);

    float eigenvalue = 0.0f;
    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<float> newVector = matrixVectorMultiply(matrix, eigenvector);
        normalizeVector(newVector);

        float newEigenvalue = dotProduct(newVector, matrixVectorMultiply(matrix, newVector));

        if (std::fabs(newEigenvalue - eigenvalue) < tolerance) {
            eigenvalue = newEigenvalue;
            eigenvector = newVector;
            break;
        }

        eigenvalue = newEigenvalue;
        eigenvector = newVector;
    }
    return {eigenvalue, eigenvector};
}

// Function to compute the eigenvector corresponding to the given eigenvalue
std::vector<float> computeEigenvector(const std::vector<std::vector<float>>& matrix, float eigenvalue, int maxIterations = 1000, float tolerance = 1e-4) {
    size_t n = matrix.size();
    std::vector<float> eigenvector(n, 1.0f); // Start with an arbitrary vector
    normalizeVector(eigenvector);

    float diff = 1.0f;
    int iterCount = 0;
    
    while (diff > tolerance && iterCount < maxIterations) {
        std::vector<float> newVector = matrixVectorMultiply(matrix, eigenvector);
        
        // Subtract eigenvalue times the eigenvector from the result
        for (size_t i = 0; i < n; ++i) {
            newVector[i] -= eigenvalue * eigenvector[i];
        }
        
        float newNorm = 0.0f;
        for (float v : newVector) {
            newNorm += v * v;
        }
        newNorm = std::sqrt(newNorm);
        
        diff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            diff += std::fabs(newVector[i] - eigenvector[i]);
            eigenvector[i] = newVector[i] / newNorm;
        }

        iterCount++;
    }
    
    return eigenvector;
}

// Main function to compute all eigenvalues and eigenvectors
void computeEigenvaluesAndEigenvectors(std::vector<std::vector<float>> matrix) {
    size_t n = matrix.size();
    std::vector<float> eigenvalues;
    std::vector<std::vector<float>> eigenvectors;

    for (size_t k = 0; k < n; ++k) {
        auto [eigenvalue, eigenvector] = powerIteration(matrix);
        eigenvalues.push_back(eigenvalue);
        eigenvectors.push_back(eigenvector);

        // Deflate the matrix
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    // Print eigenvalues and eigenvectors
    std::cout << "Eigenvalues:\n";
    for (float ev : eigenvalues) {
        std::cout << ev << " ";
    }
    std::cout << "\nEigenvectors:\n";
    for (const auto& vec : eigenvectors) {
        for (float v : vec) {
            std::cout << v << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Example covariance matrix
    std::vector<std::vector<float>> covarianceMatrix = {
        {0.0215355, 0.0162204, 0.0128816, 0.00888937},
        {0.0162204, 0.0199768, 0.0168772, 0.0132599},
        {0.0128816, 0.0168772, 0.0210823, 0.0158803},
        {0.00888937, 0.0132599, 0.0158803, 0.0233283}
    };

    computeEigenvaluesAndEigenvectors(covarianceMatrix);
=======
#include <eigen3/Eigen/Dense>

int main() {
    // Define the matrix
    Eigen::Matrix3f matrix;
    matrix << 4.0f, 1.0f, 2.0f,
              1.0f, 3.0f, 0.0f,
              2.0f, 0.0f, 2.0f;

    // Compute the eigenvalues and eigenvectors
    Eigen::EigenSolver<Eigen::Matrix3f> solver(matrix);

    // Output the eigenvalues
    std::cout << "Eigenvalues:\n" << solver.eigenvalues() << std::endl;

    // Output the eigenvectors
    std::cout << "Eigenvectors:\n" << solver.eigenvectors() << std::endl;
>>>>>>> Stashed changes

    return 0;
}
