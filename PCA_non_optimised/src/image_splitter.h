#ifndef IMAGE_SPLITTER_H
#define IMAGE_SPLITTER_H

#include <vector>
#include <string>

// Function to split an image into N smaller images (blocks)
// imageData: 1D vector representing grayscale image data (row-major order)
// imageSize: Size of the input image (assumed to be square: width == height)
// blockSize: Size of each block (assumed to be square)
// Returns: A vector of 1D vectors, each representing a smaller image block
std::vector<std::vector<float>> split_image(const std::vector<float>& imageData, int imageSize, int blockSize);

// Function to save smaller image blocks into PNG files
// blocks: A vector of 1D vectors, each representing a smaller image block
// blockSize: Size of each block (width and height)
// outputDir: Directory to save the blocks
// File naming will follow the pattern "image_1.png", "image_2.png", etc.
void save_blocks_to_png(const std::vector<std::vector<float>>& blocks, int blockSize, const std::string& outputDir);

#endif // IMAGE_SPLITTER_H
