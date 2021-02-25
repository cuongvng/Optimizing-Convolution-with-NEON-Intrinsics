#include <iostream>
#include<vector>
#include "../include/NEON_2_SSE.h"
// #include <arm_neon.h>
#include <time.h>

using namespace std;

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

float32_t* flatten_kernel(vector<vector<float32_t>> kernel, const uint32_t &kernel_height, const uint32_t &kernel_width);

vector<vector<float32_t>> convolve_neon(vector<vector<float32_t>> input, vector<vector<float32_t>> kernel){
    // Simple single-channel convolution
    clock_t start = clock();

    uint32_t input_height = input.size();
    uint32_t input_width = input[0].size();

    // Flatten the kernel, row-major
    const uint32_t KERNEL_HEIGHT = kernel.size();
    const uint32_t KERNEL_WIDTH = kernel[0].size();
    const uint32_t N_KERNEL_PIX = kernel.size() * kernel[0].size();

    float32_t kernel_data[N_KERNEL_PIX];
    for (uint8_t r=0; r<KERNEL_HEIGHT; r++){
        float32_t* kernel_row = kernel[r].data();
        for (uint8_t c=0; c<KERNEL_WIDTH; c++){
            kernel_data[r*KERNEL_WIDTH + c] = *(kernel_row+c);
        }
    }

    // Get output shape
    uint32_t output_height = input_height - KERNEL_HEIGHT + 1;
    uint32_t output_width = input_width - KERNEL_WIDTH + 1;
    vector<vector<float32_t>> result;

    // Array to store the data of sliding window on the input
    float32_t input_window[N_KERNEL_PIX];

    for (uint32_t i=0; i<output_height; i++){
        vector<float32_t> res_row;
        for (uint32_t j=0; j<output_width; j++){
            float32_t conv = 0;

            // Get data on input window
            for (uint32_t kh=0; kh<KERNEL_HEIGHT; kh++)
                for(uint32_t kw=0; kw<KERNEL_WIDTH; kw++)
                    input_window[kh*KERNEL_WIDTH + kw] = input[i+kh][j+kw];

            /**** Apply NEON Intrinsics ****/

            for (uint32_t block4_idx=0; block4_idx<N_KERNEL_PIX/4; block4_idx++){
                // Load each pair of 4-element blocks of input and kernel into ARM registers
                float32x4_t input_reg = vld1q_f32(input_window + 4*block4_idx);
                float32x4_t kernel_reg = vld1q_f32(kernel_data + 4*block4_idx);

                // Perform element-wise multiplication on the registers
                float32x4_t ew_mul_reg = vmulq_f32(input_reg, kernel_reg);

                // Load `ew_mul_reg` result from the registers back to the 4-element array `ew_mul_mem` on memory
                float32_t ew_mul_mem[4];
                vst1q_f32(ew_mul_mem, ew_mul_reg);

                // Accumulate the convolution results on the current block pairs
                for (uint8_t m=0; m<4; m++)
                    conv += ew_mul_mem[m];
            }

            // Handle the rest (N_KERNEL_PIX % 4) elements separately
            uint8_t n_rest = N_KERNEL_PIX % 4;
            if (n_rest > 0)
                for (uint8_t l=0; l<n_rest; l++)
                    conv += input_window [N_KERNEL_PIX - N_KERNEL_PIX%4 +l] * kernel_data[N_KERNEL_PIX - N_KERNEL_PIX%4 +l];

            res_row.push_back(conv);
        }
        result.push_back(res_row);
    }

    clock_t duration = clock() - start;
    cout << "Time consumed: " << float(duration)/CLOCKS_PER_SEC << endl;
    
    return result;
}

float32_t* flatten_kernel(vector<vector<float32_t>> kernel, const uint32_t &kernel_height, const uint32_t &kernel_width){
    static float32_t* flattened_kernel = new float32_t[kernel_height * kernel_width];
    
    for (uint8_t r=0; r<kernel_height; r++){
        float32_t* kernel_row = kernel[r].data();
        for (uint8_t c=0; c<kernel_width; c++){
            flattened_kernel[r*kernel_width + c] = *(kernel_row+c);
        }
    }
    return flattened_kernel;
}
