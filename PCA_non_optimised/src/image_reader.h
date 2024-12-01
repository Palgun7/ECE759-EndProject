#ifndef IMAGE_READER_H
#define IMAGE_READER_H

#include <vector>

// Function to read a PNG file and convert it to a floating point array
std::vector<float> read_png_file(const char* file_name, int& width, int& height);

// Function to print the image data
void print_image(const std::vector<float>& raw_image, int width, int height);

#endif // IMAGE_READER_H
