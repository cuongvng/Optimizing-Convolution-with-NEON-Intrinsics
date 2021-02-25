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

float32_t* flatten_kernel(vector<vector<float32_t>> kernel);

vector<vector<float32_t>> convolve_neon(vector<vector<float32_t>> input, vector<vector<float32_t>> kernel){
    // Simple single-channel convolution
    clock_t start = clock();

    uint32_t input_height = input.size();
    uint32_t input_width = input[0].size();

    uint32_t output_height = input_height - KERNEL_HEIGHT + 1;
    uint32_t output_width = input_width - KERNEL_WIDTH + 1;

    vector<vector<float32_t>> result;

    // Flatten the kernel, row-major
    float32_t* kernel_data = flatten_kernel(kernel);

    float32_t input_window[KERNEL_HEIGHT*KERNEL_WIDTH];
    for (uint32_t i=0; i<output_height; i++){
        vector<float32_t> res_row;
        for (uint32_t j=0; j<output_width; j++){
            float32_t conv = 0;

            // Get data on input window
            for (uint32_t kh=0; kh<KERNEL_HEIGHT; kh++)
                for(uint32_t kw=0; kw<KERNEL_WIDTH; kw++)
                    input_window[kh*KERNEL_WIDTH + kw] = input[i+kh][j+kw];

            /**** Apply NEON Intrinsics ****/
            // For 3x3 kernel, load 2 four-element blocks out of 9 elements in the kernel and input array to ARM registers
            // to perform multiplication on registers (expected to be faster!)
            // The last elements will be handled separately.

            float32x4_t input_reg1 = vld1q_f32(input_window);
            float32x4_t kernel_reg1 = vld1q_f32(kernel_data);
            float32x4_t input_reg2 = vld1q_f32(input_window+4);
            float32x4_t kernel_reg2 = vld1q_f32(kernel_data+4);

            float32_t input_last = input_window[8];
            float32_t kernel_last = kernel_data[8];

            // Element-wise multiplication of `input_reg` and `kernel_reg`
            float32x4_t ew_mul_reg1 = vmulq_f32(input_reg1, kernel_reg1);
            float32x4_t ew_mul_reg2 = vmulq_f32(input_reg2, kernel_reg2);

            // Load `ew_mul_reg` from register back to memory
            float32_t ew_mul_mem1[4];
            float32_t ew_mul_mem2[4];
            vst1q_f32(ew_mul_mem1, ew_mul_reg1);
            vst1q_f32(ew_mul_mem2, ew_mul_reg2);

            for (uint8_t m=0; m<4; m++)
                conv += ew_mul_mem1[m] + ew_mul_mem2[m];
            
            // Add the result of the last element
            conv += input_last * kernel_last;

            res_row.push_back(conv);
        }
        result.push_back(res_row);
    }

    clock_t duration = clock() - start;
    cout << "Time consumed: " << float(duration)/CLOCKS_PER_SEC << endl;
    
    return result;
}

float32_t* flatten_kernel(vector<vector<float32_t>> kernel){
    static float32_t flattened_kernel[KERNEL_HEIGHT*KERNEL_WIDTH];
    
    for (uint8_t r=0; r<KERNEL_HEIGHT; r++){
        float32_t* kernel_row = kernel[r].data();
        for (uint8_t c=0; c<KERNEL_WIDTH; c++){
            flattened_kernel[r*KERNEL_WIDTH + c] = *(kernel_row+c);
        }
    }
    return flattened_kernel;
}
