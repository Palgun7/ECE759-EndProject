#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using namespace Eigen;
using namespace cv;
using namespace std;

// PCA Function
pair<MatrixXd, MatrixXd> pca(const MatrixXd& X, int num_components) {
    // Step 1: Center the data
    RowVectorXd mean = X.colwise().mean();
    MatrixXd X_meaned = X.rowwise() - mean;

    // Step 2: Compute the covariance matrix
    MatrixXd covariance_matrix = (X_meaned.transpose() * X_meaned) / (X.rows() - 1);

    // Step 3: Eigenvalue decomposition
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(covariance_matrix);
    VectorXd eigenvalues = eigen_solver.eigenvalues();
    MatrixXd eigenvectors = eigen_solver.eigenvectors();

    // Step 4: Sort eigenvalues and eigenvectors in descending order
    vector<pair<double, VectorXd>> eigen_pairs;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigen_pairs.emplace_back(eigenvalues(i), eigenvectors.col(i));
    }
    sort(eigen_pairs.rbegin(), eigen_pairs.rend(),
         [](const pair<double, VectorXd>& a, const pair<double, VectorXd>& b) {
             return a.first < b.first;
         });

    // Step 5: Select the top 'num_components' eigenvectors
    MatrixXd principal_components(X.cols(), num_components);
    for (int i = 0; i < num_components; ++i) {
        principal_components.col(i) = eigen_pairs[i].second;
    }

    // Step 6: Project data onto principal components
    MatrixXd X_reduced = X_meaned * principal_components;

    // Step 7: Reconstruct the data
    MatrixXd X_reconstructed = (X_reduced * principal_components.transpose()).rowwise() + mean;

    return {X_reduced, X_reconstructed};
}

// Function to display and save original vs reconstructed images
void display_and_save_images(const MatrixXd& original, const MatrixXd& reconstructed, int n_images, int image_size, const string& output_dir) {
    for (int i = 0; i < n_images; ++i) {
        // Reshape row into image
        Mat original_image(image_size, image_size, CV_64F, const_cast<double*>(original.row(i).data()));
        Mat reconstructed_image(image_size, image_size, CV_64F, const_cast<double*>(reconstructed.row(i).data()));

        // Normalize to [0, 255] for visualization
        original_image.convertTo(original_image, CV_8U, 255.0);
        reconstructed_image.convertTo(reconstructed_image, CV_8U, 255.0);

        // Save images
        string original_filename = output_dir + "/original_" + to_string(i) + ".png";
        string reconstructed_filename = output_dir + "/reconstructed_" + to_string(i) + ".png";
        imwrite(original_filename, original_image);
        imwrite(reconstructed_filename, reconstructed_image);

        // Display images (optional)
        imshow("Original Image", original_image);
        imshow("Reconstructed Image", reconstructed_image);
        waitKey(0); // Wait for key press to proceed
    }
}

int main() {
    // Load dataset (MNIST example; replace with your dataset)
    // Assume data is loaded as a MatrixXd with shape (n_samples, n_features)
    int num_samples = 100;   // Example: 100 images
    int num_features = 28 * 28; // Example: each image is 28x28 pixels
    int num_components = 50; // Retain top 50 components
    int image_size = 28;     // Image dimensions (28x28)
    string output_dir = "output_images"; // Directory to save images

    // Generate dummy data (replace with actual MNIST loading logic)
    MatrixXd data = MatrixXd::Random(num_samples, num_features);

    // Perform PCA
    auto [X_reduced, X_reconstructed] = pca(data, num_components);

    // Display and save images
    display_and_save_images(data, X_reconstructed, 5, image_size, output_dir);

    return 0;
}
