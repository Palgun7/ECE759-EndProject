#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <png.h>

using namespace std;

// Function to read a grayscale .img file and convert it to a 1D matrix of intensities
std::vector<float> read_grayscale_img(const std::string& fileName, int width, int height) {
    // Open the file in binary mode
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open the file: " + fileName);
    }

    // Calculate the total number of pixels
    int numPixels = width * height;

    // Allocate a buffer to store the pixel data
    std::vector<unsigned char> pixelData(numPixels);

    // Read the pixel data from the file
    file.read(reinterpret_cast<char*>(pixelData.data()), numPixels);
    if (file.gcount() != numPixels) {
        throw std::runtime_error("File size does not match the expected dimensions.");
    }

    file.close();

    // Convert the pixel data to a normalized 1D matrix of float intensities (0.0 to 1.0)
    std::vector<float> intensityMatrix(numPixels);
    for (int i = 0; i < numPixels; ++i) {
        intensityMatrix[i] = pixelData[i]/255.0f; // Normalize to [0, 1]
    }

    return intensityMatrix;
}

void write_png(const std::string& fileName, const std::vector<float>& intensityMatrix, int width, int height) {
    // Open the file for writing
    FILE* fp = fopen(fileName.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Failed to open file for writing: " + fileName);
    }

    // Create the png struct and info struct
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("Failed to create PNG write struct");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        throw std::runtime_error("Failed to create PNG info struct");
    }

    // Set up error handling
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw std::runtime_error("Error during PNG creation");
    }

    // Set up the output stream for the PNG file
    png_init_io(png, fp);

    // Set the image properties
    png_set_IHDR(
        png, info, width, height, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
);

    // Write the header (metadata)
    png_write_info(png, info);

    // Allocate a row buffer to hold the pixel data
    std::vector<png_byte> row(3 * width);

    // Write the image data row by row
    for (int y = 0; y < height; ++y) {
        // Convert the 1D vector of intensities to RGB values for each pixel
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float intensity = intensityMatrix[idx];
            unsigned char value = static_cast<unsigned char>(intensity * 255.0f); // Convert to 0-255 range

            // Assuming grayscale intensity, set R, G, and B to the same value
            row[3 * x] = value;        // Red
            row[3 * x + 1] = value;    // Green
            row[3 * x + 2] = value;    // Blue
        }

        // Write the row of pixel data
        png_write_row(png, row.data());
    }

    // Finish writing the PNG file
    png_write_end(png, nullptr);

    // Clean up and close the file
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

int main() {
    try {
        // Example usage
        std::string fileName = "walk_small.png";
        int width = 100; // Image width
        int height = 100; // Image height

        // Read the .img file and get the 1D intensity matrix
        std::vector<float> intensities = read_grayscale_img(fileName, width, height);

        // Print some of the intensities for demonstration
        for (int i = 0; i < 100; i++)
        {
            for (int i = 0; i < 100; ++i) {
            std::cout << intensities[i] << " ";
        }
        std::cout << std::endl;

        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;

    return 0;
}
