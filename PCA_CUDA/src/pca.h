#ifndef PCA_H
#define PCA_H

#include <vector>

// Function to compute PCA (implemented in the .cu file)
std::vector<std::vector<float>> computePCA(const std::vector<std::vector<float>> &matrix);

#endif // PCA_H
