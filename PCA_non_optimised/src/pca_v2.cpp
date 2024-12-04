#include "image_reader.h"
#include "image_normalizer.h"
#include "covariance_matrix.h"
#include "eigen_decomposition.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    const char* file_name = "../data/Solid_white.png";
    int width, height;

    try {
        std::vector<float> raw_image = read_png_file(file_name, width, height);
        std::cout << "Image loaded successfully. Width: " << width << ", Height: " << height << std::endl;

        // Normalize the grayscale image
        std::vector<float> normalized_image = normalize_image(raw_image);
        
	for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
		std::cout << normalized_image[i * width + j] << " ";  // Access element in row-major order
        }
	std::cout << '\n';
    }
	
	// Compute covariance
	std::vector<std::vector<float>> covariance_matrix = compute_covariance_matrix(normalized_image, width, height);

        // Print the covariance matrix
        std::cout << "Covariance Matrix:\n";
        for (const auto& row : covariance_matrix) {
            for (const auto& value : row) {
                std::cout << std::fixed << std::setprecision(4) << value << " ";
            }
            std::cout << "\n";
        }
	// for sanity check
	return 0;
	// Compute eigenvalues and eigenvectors
        std::vector<float> eigenvalues;
        std::vector<std::vector<float>> eigenvectors;
        compute_eigenvalues_and_eigenvectors(covariance_matrix, eigenvalues, eigenvectors);

        // Display eigenvalues
        std::cout << "Eigenvalues:\n";
        for (const auto& value : eigenvalues) {
            std::cout << std::fixed << std::setprecision(4) << value << " ";
        }
        std::cout << "\n";

        // Create and display the feature vector
        std::vector<float> feature_vector = create_feature_vector(normalized_image, eigenvectors, width, height);
        std::cout << "Feature Vector:\n";
        for (const auto& value : feature_vector) {
            std::cout << std::fixed << std::setprecision(4) << value << " ";
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
