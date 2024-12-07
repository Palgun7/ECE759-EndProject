#ifndef POWER_ITERATION_H
#define POWER_ITERATION_H

#include <vector>

// Function to compute eigenvectors using power iteration
std::vector<std::vector<float>> powerIteration(const std::vector<std::vector<float>>& CovarianceMatrix, int maxIterations = 1000, float tolerance = 1e-6);

#endif // POWER_ITERATION_H
