#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <rows> <cols> <input_file> <output_image>" << std::endl;
        return -1;
    }

    // Read command-line arguments
    int rows = std::stoi(argv[1]);
    int cols = std::stoi(argv[2]);
    std::string inputFilePath = argv[3];
    std::string outputImagePath = argv[4];

    // Validate dimensions
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: Invalid image dimensions." << std::endl;
        return -1;
    }

    // Open the input file for reading
    std::ifstream inFile(inputFilePath);
    if (!inFile.is_open()) {
        std::cerr << "Error: Unable to open input file at " << inputFilePath << std::endl;
        return -1;
    }

    // Read all pixel values from the file into a single line
    std::string line;
    std::getline(inFile, line);
    inFile.close();

    // Parse pixel values
    std::istringstream iss(line);
    std::vector<int> pixelValues;
    int value;
    while (iss >> value) {
        if (value < 0 || value > 255) {
            std::cerr << "Error: Invalid pixel value in input file." << std::endl;
            return -1;
        }
        pixelValues.push_back(value);
    }

    // Ensure the number of pixels matches the expected dimensions
    if (pixelValues.size() != rows * cols) {
        std::cerr << "Error: Mismatch between number of pixel values and specified dimensions." << std::endl;
        return -1;
    }

    // Create an empty grayscale image
    cv::Mat image(rows, cols, CV_8UC1);

    // Fill the image matrix with pixel values in row-major order
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            image.at<uchar>(i, j) = static_cast<uchar>(pixelValues[i * cols + j]);
        }
    }

    // Save the reconstructed image
    if (!cv::imwrite(outputImagePath, image)) {
        std::cerr << "Error: Unable to save the image at " << outputImagePath << std::endl;
        return -1;
    }

    std::cout << "Image reconstructed and saved to " << outputImagePath << std::endl;

    return 0;
}
