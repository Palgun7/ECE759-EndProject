#ifndef IMAGE_READER_H
#define IMAGE_READER_H

#include <vector>

// Function to read a PNG file (grayscale or RGB) and convert it to grayscale
std::vector<float> read_png_file(const char* file_name, int& width, int& height);

#endif // IMAGE_READER_H
