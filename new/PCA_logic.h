#ifndef PCA_LOGIC_H
#define PCA_LOGIC_H

#include <vector>
#include <utility>

// Sorts eigenvalues and eigenvectors in descending order of eigenvalues
std::pair<std::vector<double>, std::vector<std::vector<double>>>
sortEigenvaluesAndVectors(const std::vector<double>& eigenvalues, const std::vector<std::vector<double>>& eigenvectors);

// Selects top-k eigenvectors
std::vector<std::vector<double>> getTopKEigenvectors(const std::vector<std::vector<double>>& eigenvectors, int k);

// Projects data to principal components
std::vector<std::vector<double>> projectToPrincipalComponents(
    const std::vector<std::vector<double>>& normalizedMatrix,
    const std::vector<std::vector<double>>& principalComponents);

std::vector<std::vector<double>> computeCovariance(const std::vector<std::vector<double>>& matrix);
std::pair<std::vector<double>, std::vector<std::vector<double>>> computeEigen(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> projectData(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& eigenvectors);

#endif // PCA_H
