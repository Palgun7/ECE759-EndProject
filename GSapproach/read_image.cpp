#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

bool isGrayscale(const cv::Mat& image) {
    // Check if the image has three channels and all channel values are identical
    if (image.channels() == 3) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                if (pixel[0] != pixel[1] || pixel[1] != pixel[2]) {
                    return false; // Not grayscale
                }
            }
        }
        return true; // All channel values are identical
    }
    return false; // Not three-channel image
}

int main(int argc, char* argv[]) {
    // Check for proper usage
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <output_file>" << std::endl;
        return -1;
    }

    // Read command-line arguments
    std::string imagePath = argv[1];
    std::string outputFilePath = argv[2];

    // Read the image (PNG or any other format supported by OpenCV)
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_UNCHANGED);

    // Check if the image is loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Unable to load image at " << imagePath << std::endl;
        return -1;
    }

    // Open the output file for writing
    std::ofstream outFile(outputFilePath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open output file at " << outputFilePath << std::endl;
        return -1;
    }

    // Write image dimensions and number of channels
    outFile << "Image Dimensions: " << image.rows << "x" << image.cols << std::endl;
    outFile << "Number of Channels: " << image.channels() << std::endl;

    // Check if the image is grayscale (single channel or 3-channel with identical values)
    if (image.channels() == 1 || isGrayscale(image)) {
        outFile << "Image is Grayscale. Pixel values in row-major order:" << std::endl;
        if (image.channels() == 1) { // Single channel grayscale image
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    outFile << (int)image.at<uchar>(i, j) << " ";
                }
            }
        } else { // 3-channel grayscale image
            for (int i = 0; i < image.rows; ++i) {
                for (int j = 0; j < image.cols; ++j) {
                    cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                    outFile << (int)pixel[0] << " "; // Write one channel (all are identical)
                }
            }
        }
        outFile << std::endl;
    } else {
        outFile << "Image is not Grayscale. Writing matrix values (BGR format):" << std::endl;
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                outFile << "(" << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << ") ";
            }
            outFile << std::endl;
        }
    }

    // Close the output file
    outFile.close();

    std::cout << "Image matrix saved to " << outputFilePath << std::endl;

    return 0;
}
