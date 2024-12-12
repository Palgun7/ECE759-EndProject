#include <iostream>
#include <vector>
#include <png.h>                    // to parse png image
#include <stdexcept>                // to raise exceptions
#include <cstdio>
#include <cmath>                    // for sqrt function
#include <fstream>                  // for file operations
using namespace std;

/////////////////// Compilation Instruction /////////////////////
// g++ get_image_matrix.cpp -o get_image_matrix -std=c++17 -lpng
// ./get_image_matrix <image_file> > <output_file>
// eg. ./get_image_matrix man.png >output.txt
/////////////////////////////////////////////////////////////////

// Function to read a PNG file (grayscale or RGB) and convert it to grayscale
// returns 1D vector of intensities
vector<float> read_png_file(const char* file_name, int& width, int& height) {
    FILE* fp = fopen(file_name, "rb");
    if (!fp) {
        throw std::runtime_error("Error: Unable to open file " + std::string(file_name));
    }

    // Read PNG file signature
    png_byte header[8];
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8)) {
        fclose(fp);
        throw std::runtime_error("Error: File is not recognized as a PNG");
    }

    // Initialize PNG structures
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        throw std::runtime_error("Error: Failed to create PNG read structure");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        throw std::runtime_error("Error: Failed to create PNG info structure");
    }

    // Error handling
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        throw std::runtime_error("Error: libpng encountered an error");
    }

    // Set up PNG file reading
    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    // Get image details
    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);
    png_byte color_type = png_get_color_type(png, info);

    // Convert palette/grayscale images to RGB/RGBA
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (bit_depth == 16)
        png_set_strip_16(png);

    // Update information after transformations
    png_read_update_info(png, info);

    // Get the number of channels (e.g., 3 for RGB, 4 for RGBA)
    int channels = png_get_channels(png, info);

    // Allocate memory for the image rows
    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
    }

    // Read the image
    png_read_image(png, row_pointers.data());

    // Convert to a flat float vector (row-major order)
    std::vector<float> float_data(width * height * channels);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width * channels; x++) {
            float_data[y * width * channels + x] = row_pointers[y][x] / 255.0f;
        }
    }

    // Clean up
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return float_data;
}

// Function to split an image into N smaller blocks
// returns 2D vector of image-blocks stacked on top of each other
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

// Function to print a matrix
void printMatrix(const vector<vector<float>> &matrix)
{
    for (const auto &row : matrix)
    {
        for (float val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }
}

// Function to write a matrix to a text file
void writeMatrixToFile(const string &filename, const vector<vector<float>> &matrix)
{
    ofstream file(filename);
    if (file.is_open())
    {
        for (const auto &row : matrix)
        {
            for (float val : row)
            {
                file << val << " ";
            }
            file << endl;
        }
        file.close();
    }
    else
    {
        cerr << "Unable to open file " << filename << endl;
    }
}

int main(int argc, char* argv[])
{
    // get the file path as a command line argument
    char *inputFileName = argv[1];
    char *outputFileName = argv[2];
    
    // parse the image file 
    int imageWidth, imageHeight;

    // Read the image
    auto imageData = read_png_file(inputFileName, imageWidth, imageHeight);

    // Ensure the image is square 
    if (imageWidth != imageHeight) {
        cerr << "Error: Image must be square." << endl;
        return -1;
    }

    int blockSize = sqrt(imageHeight);

    // Split the image into smaller blocks
    auto blocks = split_image(imageData, imageWidth, blockSize);

    // Write the matrix to a text file
    writeMatrixToFile(outputFileName, blocks);

    // Print the matrix
    printMatrix(blocks);

    // Print the dimensions
    cout << "Rows: " << blocks.size() << endl;
    cout << "Columns: " << blocks[0].size() << endl;

    return 0;
}
