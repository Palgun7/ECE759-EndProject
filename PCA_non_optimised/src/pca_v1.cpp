#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

// Function to generate a random matrix
vector<vector<double>> generateRandomMatrix(int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// Function to print a matrix
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// Function to compute the transpose of a matrix
vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> transposed(cols, vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

// Function to multiply two matrices
vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows = A.size();
    int cols = B[0].size();
    int inner = A[0].size();

    vector<vector<double>> result(rows, vector<double>(cols, 0.0));

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
vector<double> computeColumnMeans(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<double> means(cols, 0.0);

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            means[j] += matrix[i][j];
        }
        means[j] /= rows;
    }
    return means;
}

// Function to center the matrix
vector<vector<double>> centerMatrix(const vector<vector<double>>& matrix, const vector<double>& means) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> centered(rows, vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            centered[i][j] = matrix[i][j] - means[j];
        }
    }
    return centered;
}

// Function to perform power iteration to find the largest eigenvalue and eigenvector
pair<double, vector<double>> powerIteration(const vector<vector<double>>& matrix, int maxIter = 1000, double tol = 1e-6) {
    int n = matrix.size();
    vector<double> b(n, 1.0); // Initial vector
    double eigenvalue = 0.0;

    for (int iter = 0; iter < maxIter; ++iter) {
        vector<double> b_next(n, 0.0);

        // Multiply matrix with b
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                b_next[i] += matrix[i][j] * b[j];
            }
        }

        // Normalize b_next
        double norm = 0.0;
        for (double val : b_next) {
            norm += val * val;
        }
        norm = sqrt(norm);
        for (double& val : b_next) {
            val /= norm;
        }

        // Check for convergence
        double diff = 0.0;
        for (int i = 0; i < n; ++i) {
            diff += abs(b_next[i] - b[i]);
        }
        if (diff < tol) {
            // Compute eigenvalue
            eigenvalue = 0.0;
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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <rows> <cols>" << endl;
        return 1;
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);

    if (rows <= 0 || cols <= 0) {
        cerr << "Matrix dimensions must be positive integers!" << endl;
        return 1;
    }

    // Generate random matrix
    auto data = generateRandomMatrix(rows, cols);
    cout << "Randomly generated matrix:\n";
    printMatrix(data);

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

    cout << "\nLargest Eigenvalue: " << eigenvalue << endl;
    cout << "Corresponding Eigenvector:\n";
    for (double val : eigenvector) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
