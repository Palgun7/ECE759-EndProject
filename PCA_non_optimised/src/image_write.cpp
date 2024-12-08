#include "image_write.h"
#include <cstdio>
#include <cstdlib>

// Function to write a PNG file from image data
void write_png_file(const char* file_name, const std::vector<float>& image_data, int width, int height) {
    FILE* fp = fopen(file_name, "wb");
    if (!fp) {
        throw std::runtime_error("Failed to open file for writing: " + std::string(file_name));
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("Failed to create PNG write struct.");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        throw std::runtime_error("Failed to create PNG info struct.");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw std::runtime_error("Error during PNG creation.");
    }

    png_init_io(png, fp);

    // Set image properties
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Allocate memory for row pointers and fill them with image data
    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
        for (int x = 0; x < width; x++) {
            row_pointers[y][x] = static_cast<png_byte>(image_data[y * width + x] * 255.0f);
        }
    }

    // Write the image data to the PNG file
    png_write_image(png, row_pointers.data());
    png_write_end(png, NULL);

    // Free the allocated memory
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}
