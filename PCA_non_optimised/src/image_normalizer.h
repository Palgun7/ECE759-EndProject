#ifndef IMAGE_NORMALIZER_H
#define IMAGE_NORMALIZER_H

#include <vector>

// Function to normalize an image for PCA
std::vector<float> normalize_image(std::vector<float>& image, int width, int height);

#endif // IMAGE_NORMALIZER_H
