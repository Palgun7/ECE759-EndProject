#include "image_normalizer.h"
#include <numeric>
#include <cmath>
#include <vector>

std::vector<float> normalize_image(const std::vector<float>& image) {
    int size = image.size();

    // Calculate mean
    float mean = std::accumulate(image.begin(), image.end(), 0.0f) / size;

    // Calculate variance (sum of squared differences from mean)
    float sq_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sq_sum += (image[i] - mean) * (image[i] - mean);
    }

    // Calculate standard deviation
    float stddev = std::sqrt(sq_sum / size);  // If you want sample standard deviation, use (size - 1)				
    std::vector<float> normalized_image(size);
      
    if (stddev == 0){
	    stddev = 1;}

    // Normalize the image
        for (int i = 0; i < size; ++i) {
        normalized_image[i] = (image[i] - mean) / stddev;
    }

    return normalized_image;
}
