#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <png.h>

using namespace std;

// Function to read a PNG file and return the image data as a vector of floats
vector<float> read_png_file(const char* file_name, int& width, int& height) {
    FILE* fp = fopen(file_name, "rb");
    if (!fp) {
        throw runtime_error("Failed to open file");
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        throw runtime_error("Failed to create png read struct");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        throw runtime_error("Failed to create png info struct");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        throw runtime_error("Error during png read");
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) {
        png_set_strip_16(png);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA) {
        png_set_rgb_to_gray_fixed(png, 1, -1, -1); // Convert RGB to grayscale
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }

    png_read_update_info(png, info);

    vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers.data());

    vector<float> image_data(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image_data[y * width + x] = row_pointers[y][x] / 255.0f;
        }
    }

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return image_data;
}

// Function to write a PNG file from image data
void write_png_file(const char* file_name, const vector<float>& image_data, int width, int height) {
    FILE* fp = fopen(file_name, "wb");
    if (!fp) {
        throw runtime_error("Failed to open file for writing");
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        throw runtime_error("Failed to create png write struct");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        throw runtime_error("Failed to create png info struct");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        throw runtime_error("Error during png write");
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
        for (int x = 0; x < width; x++) {
            row_pointers[y][x] = static_cast<png_byte>(image_data[y * width + x] * 255.0f);
        }
    }

    png_write_image(png, row_pointers.data());
    png_write_end(png, NULL);

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

// Function to print a matrix
void printMatrix(const vector<vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// Function to compute the transpose of a matrix
vector<vector<float>> transpose(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<float>> transposed(cols, vector<float>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Function to multiply two matrices
vector<vector<float>> multiply(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    int rows = A.size();
    int cols = B[0].size();
    int inner = A[0].size();

    vector<vector<float>> result(rows, vector<float>(cols, 0.0f));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < inner; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Function to compute the mean of each column
vector<float> computeColumnMeans(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<float> means(cols, 0.0f);

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            means[j] += matrix[i][j];
        }
        means[j] /= rows;
    }
    return means;
}

// Function to center the matrix
vector<vector<float>> centerMatrix(const vector<vector<float>>& matrix, const vector<float>& means) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<float>> centered(rows, vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            centered[i][j] = matrix[i][j] - means[j];
        }
    }
    return centered;
}

// Function to perform power iteration to find the largest eigenvalue and eigenvector
pair<float, vector<float>> powerIteration(const vector<vector<float>>& matrix, int maxIter = 10000000, float tol = 1e-7) {
    int n = matrix.size();
    vector<float> b(n, 1.0f); // Initial vector
    float eigenvalue = 0.0f;

    for (int iter = 0; iter < maxIter; ++iter) {
	if(iter%1000 == 0) 
		cout << "|";
        vector<float> b_next(n, 0.0f);

        // Multiply matrix with b
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                b_next[i] += matrix[i][j] * b[j];
            }
        }

        // Normalize b_next
        float norm = 0.0f;
        for (float val : b_next) {
            norm += val * val;
        }
        norm = sqrt(norm);
        for (float& val : b_next) {
            val /= norm;
        }

        // Check for convergence
        float diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            diff += abs(b_next[i] - b[i]);
        }
        if (diff < tol) {
            // Compute eigenvalue
            eigenvalue = 0.0f;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    eigenvalue += b_next[i] * matrix[i][j] * b_next[j];
                }
            }
            return {eigenvalue, b_next};
        }
        b = b_next;
    }

    cerr << "Power iteration did not converge!" << endl;
    return {eigenvalue, b};
}



// Function to project data onto the principal components
vector<vector<float>> projectOntoPrincipalComponents(const vector<vector<float>>& centered, const vector<vector<float>>& eigenvectors) {
    int rows = centered.size();
    int numComponents = eigenvectors.size();
    vector<vector<float>> projected(rows, vector<float>(numComponents, 0.0f));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < numComponents; ++j) {
            for (int k = 0; k < centered[0].size(); ++k) {
                projected[i][j] += centered[i][k] * eigenvectors[j][k];
            }
        }
    }
    return projected;
}

// Function to reconstruct data from the principal components
vector<vector<float>> reconstructFromPrincipalComponents(const vector<vector<float>>& projected, const vector<vector<float>>& eigenvectors, const vector<float>& means) {
    int rows = projected.size();
    int cols = eigenvectors[0].size();
    vector<vector<float>> reconstructed(rows, vector<float>(cols, 0.0f));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < projected[0].size(); ++k) {
                reconstructed[i][j] += projected[i][k] * eigenvectors[k][j];
            }
            reconstructed[i][j] += means[j];
        }
    }
    return reconstructed;
}

int main(int argc, char* argv[]) {

    int rows, cols;
    const char* inputFileName = "../data/flower_small.png";
    const char* outputFileName = "../data/output_image.png";

    // Read image data
    auto image_data = read_png_file(inputFileName, rows, cols);

    // Convert image data to matrix form
    vector<vector<float>> data(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = image_data[i * cols + j];
        }
    }


    // Center the data
    auto means = computeColumnMeans(data);
    auto centered = centerMatrix(data, means);

    // Compute covariance matrix
    auto transposed = transpose(centered);
    auto covariance = multiply(transposed, centered);

    cout << "\nCovariance matrix:\n";
    printMatrix(covariance);

    // Perform power iteration for the largest eigenvalue and eigenvector
    auto [eigenvalue, eigenvector] = powerIteration(covariance);

    // Convert eigenvector to matrix form
    vector<vector<float>> eigenvectors = {eigenvector};

    // Project the centered data onto the principal components
    auto projected = projectOntoPrincipalComponents(centered, eigenvectors);

    // Reconstruct the image from the principal components
    auto reconstructed = reconstructFromPrincipalComponents(projected, eigenvectors, means);

    // Convert reconstructed matrix to image data
    vector<float> output_image_data(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output_image_data[i * cols + j] = reconstructed[i][j];
        }
    }

    // Write the output image
    write_png_file(outputFileName, output_image_data, cols, rows);

    cout << "Output image saved to " << outputFileName << endl;

    return 0;
}
