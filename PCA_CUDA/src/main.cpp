#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include "pca.h" // Include the PCA header file

using namespace std;

// Function to read a 2D matrix from a text file
vector<vector<float>> readMatrixFromFile(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(1);
    }

    vector<vector<float>> matrix;
    string line;
    while (getline(file, line)) {
        vector<float> row;
        stringstream ss(line);
        float value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    return matrix;
}

// Function to write a 2D matrix to a text file
void writeMatrixToFile(const string &filename, const vector<vector<float>> &matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(1);
    }

    for (const auto &row : matrix) {
        for (float val : row) {
            file << val << " ";
        }
        file << endl;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];

    // Read the matrix from the file
    auto matrix = readMatrixFromFile(inputFile);

    // Start timing
    auto start = chrono::high_resolution_clock::now();

    // Perform PCA using CUDA
    vector<vector<float>> projectedMatrix = computePCA(matrix);

    // End timing
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "PCA computation completed in " << elapsed.count() << " seconds." << endl;

    // Write the projected matrix to the output file
    writeMatrixToFile(outputFile, projectedMatrix);

    return 0;
}
