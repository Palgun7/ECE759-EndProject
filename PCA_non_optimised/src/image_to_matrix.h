#ifndef IMAGE_TO_MATRIX_H
#define IMAGE_TO_MATRIX_H

#include <vector>
#include <string>

// Function declaration
std::vector<std::vector<float>> createImageMatrix(const std::string& directoryPath, int& imageWidth, int& imageHeight);

#endif // IMAGE_TO_MATRIX_H
