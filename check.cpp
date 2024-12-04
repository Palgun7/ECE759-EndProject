#include <iostream>
#include <vector>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include "standardize_matrix.h"
#include "covariance_matrix.h"
#include "eigen_decomposition.h"

// Helper function to print a matrix
void print_matrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "\n";
    }
}

// Helper function to print Eigen matrices
void print_eigen_matrix(const Eigen::MatrixXd& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << matrix(i, j) << " ";
        }
        std::cout << "\n";
    }
}

// Helper function to print Eigen vectors
void print_eigen_vector(const Eigen::VectorXd& vec) {
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << vec[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    // Input matrix
    std::vector<std::vector<double>> X = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 10.0},
        {9.0, 10.0, 15.0}
    };

    // Standardize the matrix
    std::vector<std::vector<double>> standardized_X = standardize_matrix(X);
    std::cout << "Standardized Matrix:\n";
    print_matrix(standardized_X);

    // Compute covariance matrix using custom implementation
    auto covariance_matrix = compute_covariance_matrix(standardized_X);
    std::cout << "\nCustom Covariance Matrix:\n";
    print_matrix(covariance_matrix);

    // Eigenvalue decomposition using custom implementation
    auto [custom_eigenvalues, custom_eigenvectors] = eigen_decomposition(covariance_matrix);

    std::cout << "\nCustom Eigenvalues:\n";
    for (double eigenvalue : custom_eigenvalues) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << eigenvalue << " ";
    }
    std::cout << "\n\nCustom Eigenvectors:\n";
    print_matrix(custom_eigenvectors);

    // Eigen library computations
    Eigen::MatrixXd eigen_X(X.size(), X[0].size());
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[0].size(); ++j) {
            eigen_X(i, j) = X[i][j];
        }
    }

    // Standardize using Eigen
    Eigen::MatrixXd standardized_eigen_X = eigen_X.rowwise() - eigen_X.colwise().mean();
    Eigen::MatrixXd eigen_covariance_matrix = (standardized_eigen_X.transpose() * standardized_eigen_X) / (X.size() - 1);
    std::cout << "\nEigen Covariance Matrix:\n";
    print_eigen_matrix(eigen_covariance_matrix);

    // Eigenvalue decomposition using Eigen library
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(eigen_covariance_matrix);
    Eigen::VectorXd eigen_eigenvalues = eigen_solver.eigenvalues();
    Eigen::MatrixXd eigen_eigenvectors = eigen_solver.eigenvectors();

    std::cout << "\nEigen Eigenvalues:\n";
    print_eigen_vector(eigen_eigenvalues);

    std::cout << "\nEigen Eigenvectors:\n";
    print_eigen_matrix(eigen_eigenvectors);

    return 0;
}
