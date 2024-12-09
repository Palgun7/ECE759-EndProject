#include <iostream>
#include <cstdlib> // To generate random numbers
#include <random>  // To generate random numbers
#include <chrono>
#include <ratio>
#include <vector>
#include "pca.h"
using std::chrono::duration;
using std::chrono::high_resolution_clock;

using namespace std;

int main(int argc, char *argv[])
{
    // declarations needed for timer
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // image dimension
    size_t img_dim = atoi(argv[1]);
    size_t block_dim = atoi(argv[2]);

    // array to store n random float values from -1.0 to 1.0
    std::vector<std::vector<float>>arr(img_dim, std::vector<float>(img_dim, 0));

    std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source()); 
	std::uniform_real_distribution<float> dist(-1., 1.);

	// Write a random value to each slot in N
	for (int i = 0; i < (img_dim/block_dim)*(img_dim/block_dim); i++) 
    {
		for (int j = 0; j < block_dim*block_dim; j++)
        {
            arr[i][j] = dist(generator);
            // cout<<arr[i][j]<<" ";
        }
        // cout<<"\n";
	}



    ////////////////// Get the starting timestamp //////////////////
    start = high_resolution_clock::now();
    vector<float> output = pca(arr, block_dim, img_dim, img_dim);
    end = high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////

    // Convert the calculated duration to a double
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout <<"\n"<< duration_sec.count() << endl;
    
    
    

    return 0;
}