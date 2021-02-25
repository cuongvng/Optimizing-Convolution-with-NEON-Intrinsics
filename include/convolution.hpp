#include <iostream>
#include<vector>
#include <time.h>

using namespace std;

// Simple convolution function.
vector<vector<float32_t>> simply_convolve(
    vector<vector<float32_t>> input, 
    vector<vector<float32_t>> kernel){  
    // Input shape (input_height, input_width)
    // Kernel shape (kernel_height, kernel_width)

    clock_t start = clock();

    int input_height = input.size();
    int input_width = input[0].size();
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    vector<vector<float32_t>> output;
    for (int i=0; i<output_height; i++)
        output.push_back(vector<float32_t> (output_width, 0.0));

    float32_t conv = 0;

	// Fill output matrix: rows and columns are i and j respectively
    for (int i=0; i<output_height; i++){
        for (int j=0; j<output_width; j++){

            for (int k=0; k<kernel_height; k++)
                for (int l=0; l<kernel_width; l++)
                    conv += input[i+k][j+l] * kernel[k][l];
            
            output[i][j] = conv;
            conv = 0;
        }
    }

    clock_t duration = clock() - start;
    cout << "None-NEON convolution: time consumed = " << float(duration)*1e6/CLOCKS_PER_SEC << " us" << endl;

    return output;
}

// Single-channel 3D convolution function.
vector<vector<float32_t>>& convolve(
    vector<vector<vector<float32_t>>> input, 
    vector<vector<vector<float32_t>>> kernel){   
    // Input shape (channels, height, width)
    // Kernel shape (channels, height, width)

    int input_height = input[0].size();
    int input_width = input[1].size();
    int kernel_height = kernel[0].size();
    int kernel_width = kernel[1].size();

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    static vector<vector<float32_t>> output;
    for (int i=0; i<output_height; i++)
        output.push_back(vector<float32_t> (output_width, 0.0));

    float32_t res = 0; // This holds the convolution results for an index.

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

