#include <iostream>
#include <cstdlib> // To generate random numbers
#include <random>  // To generate random numbers
#include <chrono>
#include <ratio>
#include <vector>
#include <cuda.h>
#include "pca_cuda.h"
using std::chrono::duration;
using std::chrono::high_resolution_clock;

using namespace std;

int main(int argc, char *argv[])
{
   cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Get the elapsed time in milliseconds
    float ms;

    // image dimension
    size_t img_dim = atoi(argv[1]);
    size_t block_dim = atoi(argv[2]);

    int width = block_dim*block_dim;
    int height = (img_dim/block_dim)*(img_dim/block_dim);

    // array to store n random float values from -1.0 to 1.0
    std::vector<std::vector<float>>arr(width, std::vector<float>(height, 0));

    std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source()); 
	std::uniform_real_distribution<float> dist(-1., 1.);

	// Write a random value to each slot in N
	for (int i = 0; i < width; i++) 
    {
		for (int j = 0; j < height; j++)
        {
            arr[i][j] = dist(generator);
            // cout<<arr[i][j]<<" ";
        }
        // cout<<"\n";
	}


    cout<<"before calculation"<<endl;
    ////////////////// Get the starting timestamp //////////////////
    cudaEventRecord(start);
    vector<float> output = pca(arr, block_dim, img_dim, img_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ///////////////////////////////////////////////////////////////

    // Convert the calculated duration to a double
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << ms << std::endl;
    
    cout<<"after calculation"<<endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}