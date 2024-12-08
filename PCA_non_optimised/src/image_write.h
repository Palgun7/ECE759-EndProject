#ifndef IMAGE_WRITE_H
#define IMAGE_WRITE_H

#include <vector>
#include <stdexcept>
#include <png.h>

// Function to write a PNG file from image data
void write_png_file(const char* file_name, const std::vector<float>& image_data, int width, int height);

#endif // IMAGE_WRITE_H
