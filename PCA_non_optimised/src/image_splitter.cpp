#include "image_splitter.h"
#include <png.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

// Function to split an image into N smaller blocks
std::vector<std::vector<float>> split_image(const std::vector<float>& imageData, int imageSize, int blockSize) {
    if (imageSize % blockSize != 0) {
        throw std::invalid_argument("Image size must be divisible by block size.");
    }

    int numBlocks = imageSize / blockSize;
    std::vector<std::vector<float>> blocks;

    for (int rowBlock = 0; rowBlock < numBlocks; ++rowBlock) {
        for (int colBlock = 0; colBlock < numBlocks; ++colBlock) {
            std::vector<float> block(blockSize * blockSize);
            for (int i = 0; i < blockSize; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    int srcRow = rowBlock * blockSize + i;
                    int srcCol = colBlock * blockSize + j;
                    block[i * blockSize + j] = imageData[srcRow * imageSize + srcCol];
                }
            }
            blocks.push_back(block);
        }
    }
    return blocks;
}

// Helper function to write a single block to a PNG file
void write_png(const std::string& fileName, const std::vector<unsigned char>& block, int blockSize) {
    FILE* fp = fopen(fileName.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Failed to open file for writing: " + fileName);
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
    png_set_IHDR(png, info, blockSize, blockSize, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Write pixel data
    std::vector<png_bytep> row_pointers(blockSize);
    for (int y = 0; y < blockSize; ++y) {
        row_pointers[y] = (png_bytep)&block[y * blockSize];
    }
    png_write_image(png, row_pointers.data());
    png_write_end(png, NULL);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}
// Function to save image blocks to PNG files
void save_blocks_to_png(const std::vector<std::vector<float>>& blocks, int blockSize, const std::string& outputDir) {
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    for (size_t i = 0; i < blocks.size(); ++i) {
        std::ostringstream fileName;
        fileName << outputDir << "/image_" << (i + 1) << ".png";
        std::vector<unsigned char> blockByteData(blocks[i].size());
        for (size_t j = 0; j < blocks[i].size(); ++j) {
            blockByteData[j] = static_cast<unsigned char>(blocks[i][j] * 255.0f);
        }
        write_png(fileName.str(), blockByteData, blockSize);
    }
}
