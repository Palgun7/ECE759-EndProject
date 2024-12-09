#include <vector>

#ifndef PCA_CUDA_H
#define PCA_CUDA_H
// Function to read a PNG file (grayscale or RGB) and convert it to grayscale
std::vector<float> read_png_file(const char* file_name, int& width, int& height);

std::vector<float> computeColumnMeans(const std::vector<std::vector<float>> &matrix);

// perform PCA on an nxn image 
std::vector<float> pca(std::vector<std::vector<float>> imageMatrix, int blockSize, int imageWidth, int imageHeight);

#endif // PCA_H