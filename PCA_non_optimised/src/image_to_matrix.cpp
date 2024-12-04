#include "image_to_matrix.h"
#include "image_reader.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

std::vector<std::vector<float>> createImageMatrix(const std::string& directoryPath, int& imageWidth, int& imageHeight) {
    std::vector<std::vector<float>> imageMatrix;

    // Iterate over all files in the directory
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.path().extension() == ".png") { // Ensure it's a PNG file
            const std::string filePath = entry.path().string();

            // Read the PNG image as a flattened grayscale array
            std::vector<float> imageData = read_png_file(filePath.c_str(), imageWidth, imageHeight);

            // Check if the image is square (NxN)
            if (imageWidth != imageHeight) {
                std::cerr << "Error: Non-square image found at " << filePath << std::endl;
                continue;
            }

            // Add the flattened image to the matrix
            imageMatrix.push_back(imageData);
        }
    }

    return imageMatrix;
}
