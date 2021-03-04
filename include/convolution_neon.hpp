#include <iostream>
// #include "../include/neon2sse/NEON_2_SSE.h"  // Replace this include by the following to test on REAL ARM machines.
#include <arm_neon.h>

void simply_convolve_neon(
    float32_t** output, float32_t** input, float32_t** kernel,
    const uint32_t input_height, const uint32_t input_width,
    const uint32_t kernel_height, const uint32_t kernel_width){
    // Simple single-channel convolution

    // Flatten the kernel, row-major
    const uint32_t N_KERNEL_PIX = kernel_height * kernel_width;

    float32_t kernel_data[N_KERNEL_PIX];
    for (uint8_t r=0; r<kernel_height; r++){
        for (uint8_t c=0; c<kernel_width; c++){
            kernel_data[r*kernel_width + c] = kernel[r][c];
        }
    }

    // Get output shape
    uint32_t output_height = input_height - kernel_height + 1;
    uint32_t output_width = input_width - kernel_width + 1;

    // Array to store the data of sliding window on the input
    float32_t input_window[N_KERNEL_PIX];

    for (uint32_t i=0; i<output_height; i++){
        for (uint32_t j=0; j<output_width; j++){
            float32_t conv = 0;

            // Get data on input window
            for (uint32_t kh=0; kh<kernel_height; kh++)
                for(uint32_t kw=0; kw<kernel_width; kw++)
                    input_window[kh*kernel_width + kw] = input[i+kh][j+kw];

            /**** Apply NEON Intrinsics ****/

            for (uint32_t block4_idx=0; block4_idx<N_KERNEL_PIX/4; block4_idx++){
                // Load each pair of 4-element blocks of input and kernel into ARM registers
                // float32x4_t input_reg = vld1q_f32(input_window + 4*block4_idx);
                // float32x4_t kernel_reg = vld1q_f32(kernel_data + 4*block4_idx);
                float32x4_t input_reg = {
                    input_window[4*block4_idx],
                    input_window[4*block4_idx+1],
                    input_window[4*block4_idx+2],
                    input_window[4*block4_idx+3]
                };
                float32x4_t kernel_reg = {
                    kernel_data[4*block4_idx],
                    kernel_data[4*block4_idx+1],
                    kernel_data[4*block4_idx+2],
                    kernel_data[4*block4_idx+3]
                };
                // Perform element-wise multiplication on the registers
                float32x4_t ew_mul_reg = vmulq_f32(input_reg, kernel_reg);

                // // Load `ew_mul_reg` output from the registers back to the 4-element array `ew_mul_mem` on memory
                // float32_t ew_mul_mem[4];
                // vst1q_f32(ew_mul_mem, ew_mul_reg);

                // Accumulate the convolution outputs on the current block pairs
                conv += ew_mul_reg[0] + ew_mul_reg[1] + ew_mul_reg[2] + ew_mul_reg[3];
            }

            // Handle the rest (N_KERNEL_PIX % 4) elements separately
            uint8_t n_rest = N_KERNEL_PIX % 4;
            if (n_rest > 0)
                for (uint8_t l=0; l<n_rest; l++)
                    conv += input_window[N_KERNEL_PIX - N_KERNEL_PIX%4 +l] * kernel_data[N_KERNEL_PIX - N_KERNEL_PIX%4 +l];

            output[i][j] = conv;
        }
    }
}