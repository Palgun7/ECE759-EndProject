#include <vector>
using namespace std;


#ifndef PCA_H
#define PCA_H
// Function to read a PNG file (grayscale or RGB) and convert it to grayscale
std::vector<float> read_png_file(const char* file_name, int& width, int& height);

vector<float> computeColumnMeans(const vector<vector<float>> &matrix);

// perform PCA on an nxn image 
vector<float> pca(vector<vector<float>> imageMatrix, int blockSize, int imageWidth, int imageHeight);

#endif // PCA_H