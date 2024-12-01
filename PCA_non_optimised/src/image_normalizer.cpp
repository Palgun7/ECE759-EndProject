#include "image_normalizer.h"
#include <numeric>
#include <cmath>

// Function to normalize an image for PCA and return the normalized image
std::vector<float> normalize_image(std::vector<float>& image, int width, int height) {
    int size = width * height * 4; // Assuming RGBA format

    // Calculate mean
    float mean = std::accumulate(image.begin(), image.end(), 0.0f) / size;

    // Calculate standard deviation
    float sq_sum = std::inner_product(image.begin(), image.end(), image.begin(), 0.0f);
    float stddev = std::sqrt(sq_sum / size - mean * mean);

    // Create a new vector for the normalized image
    std::vector<float> normalized_image(size);

    // Normalize the image
    for (int i = 0; i < size; ++i) {
        normalized_image[i] = (image[i] - mean) / stddev;
    }

    return normalized_image;
}
