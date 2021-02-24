#include <iostream>
#include<vector>
#include "../include/NEON_2_SSE.h"
// #include <arm_neon.h>
#include <time.h>

using namespace std;

enum{
    KERNEL_HEIGHT=3,
    KERNEL_WIDTH=3
};

// Single-output 3D convolution function.
vector<vector<double>>& convolve(
    vector<vector<vector<double>>> input, 
    vector<vector<vector<double>>> kernel
    )
{   
    // Input shape (channels, height, width)
    // Kernel shape (channels, height, width)

    int input_height = input[0].size();
    int input_width = input[1].size();
    int kernel_height = kernel[0].size();
    int kernel_width = kernel[1].size();

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    static vector<vector<double>> output;
    for (int i=0; i<output_height; i++)
        output.push_back(vector<double> (output_width, 0.0));

    double res = 0; // This holds the convolution results for an index.

	// Fill output matrix: rows and columns are i and j respectively
    for (int i=0; i<output_height; i++){
        for (int j=0; j<output_width; j++){

            for (int k=0; k<kernel_height; k++)
                for (int l=0; l<kernel_width; l++)
                    for (int c=0; c<input.size(); c++)
                        res += input[c][i+k][j+l] * kernel[c][k][l];
            
            output[i][j] = res;
            res = 0;
        }
    }

    return output;
}

uint8_t* flatten_kernel(vector<vector<uint8_t>> kernel){
    static uint8_t flattened_kernel[KERNEL_HEIGHT*KERNEL_WIDTH];
    
    for (uint8_t r=0; r<KERNEL_HEIGHT; r++){
        uint8_t* kernel_row = kernel[r].data();
        for (uint8_t c=0; c<KERNEL_WIDTH; c++){
            flattened_kernel[r*KERNEL_WIDTH + c] = *(kernel_row+c);
        }
    }
    return flattened_kernel;
}
