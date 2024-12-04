#include "image_splitter.h"
#include "image_reader.h"
#include "image_to_matrix.h"
#include "center_data.h"
#include "compute_covariance.h"
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {

    int blockSize = 1;
    if (argc < 2) {
	    std::cout << "Usage : ./test <Size of the square block to be broken into>";
	    return 1;
    }
    else { 
	    blockSize = atoi(argv[1]);
    }
    const char* inputFileName = "../data/man.png";
    const std::string outputDir = "../data/dataset";
    int width, height;

    try {
        // Read the input image
        auto imageData = read_png_file(inputFileName, width, height);

        // Split the image into 10x10 blocks
        //int blockSize = 64;
        auto blocks = split_image(imageData, width, blockSize);

        // Save the blocks to PNG files
        save_blocks_to_png(blocks, blockSize, outputDir);
	int numBlocks = (height*width)/(blockSize*blockSize);

        std::cout << numBlocks << " Images split into and saved successfully!" << std::endl;

	// Create the Mx(N^2) matrix from the images
        std::vector<std::vector<float>> imageMatrix = createImageMatrix(outputDir, blockSize, blockSize);

	// Check the output
   	if (imageMatrix.empty()) {
	        std::cerr << "No valid images found in the directory." << std::endl;
     	   return -1;
	}

   	std::cout << "Number of images (M): " << imageMatrix.size() << std::endl;
        std::cout << "Image size (N x N): " << width << " x " << height << std::endl;

	 std::vector<std::vector<float>> centeredMatrix = centerData(imageMatrix);

	 std::vector<std::vector<float>> covarianceMatrix = computeCovarianceMatrix(centeredMatrix);

	 std::cout << "Covariance matrix:\n";
   	 for (const auto& row : covarianceMatrix) {
     	   for (float value : row) {
        	    std::cout << value << " ";
       	   }
           std::cout << "\n";
    	 }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}
