#include "image_reader.h"
#include <iostream>
#include <png.h>
#include <stdexcept>

// Function to read a PNG file and convert it to a floating point array
std::vector<float> read_png_file(const char* file_name, int& width, int& height) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        throw std::runtime_error("Failed to open file");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("Failed to create png read struct");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        throw std::runtime_error("Failed to create png info struct");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        throw std::runtime_error("Error during png creation");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers.data());

    std::vector<float> raw_image(width * height * 4);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row_pointers[y][x * 4]);
            raw_image[(y * width + x) * 4 + 0] = px[0] / 255.0f;
            raw_image[(y * width + x) * 4 + 1] = px[1] / 255.0f;
            raw_image[(y * width + x) * 4 + 2] = px[2] / 255.0f;
            raw_image[(y * width + x) * 4 + 3] = px[3] / 255.0f;
        }
    }

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return raw_image;
}

// Function to print the image data
void print_image(const std::vector<float>& raw_image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = raw_image[(y * width + x) * 4 + 0];
            float g = raw_image[(y * width + x) * 4 + 1];
            float b = raw_image[(y * width + x) * 4 + 2];
            std::cout << "(" << r << ", " << g << ", " << b << ") ";
        }
        std::cout << std::endl;
    }
}
