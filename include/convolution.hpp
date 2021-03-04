#include <iostream>

// Simple convolution function.
float32_t** simply_convolve_scalar(
    float32_t** input, float32_t** kernel,
    const uint32_t input_height, const uint32_t input_width,
    const uint32_t kernel_height, const uint32_t kernel_width){  

    uint32_t output_height = input_height - kernel_height + 1;
    uint32_t output_width = input_width - kernel_width + 1;

    float32_t** output = new float32_t* [output_height];

    float32_t conv = 0;

	// Fill output matrix: rows and columns are i and j respectively
    for (int i=0; i<output_height; i++){
        output[i] = new float32_t [output_width];
        for (int j=0; j<output_width; j++){

            for (int k=0; k<kernel_height; k++)
                for (int l=0; l<kernel_width; l++)
                    conv += input[i+k][j+l] * kernel[k][l];
            
            output[i][j] = conv;
            conv = 0;
        }
    }
    return output;
}

// // Single-channel 3D convolution function.
// float32_t** convolve_scalar(
//     std::vector<float32_t**> input, 
//     std::vector<float32_t**> kernel){   
//     // Input shape (channels, height, width)
//     // Kernel shape (channels, height, width)

//     int input_height = input[0].size();
//     int input_width = input[1].size();
//     int kernel_height = kernel[0].size();
//     int kernel_width = kernel[1].size();

//     int output_height = input_height - kernel_height + 1;
//     int output_width = input_width - kernel_width + 1;

//     float32_t** output;
//     for (int i=0; i<output_height; i++)
//         output.push_back(std::vector<float32_t> (output_width, 0.0));

//     float32_t res = 0; // This holds the convolution results for an index.

// 	// Fill output matrix: rows and columns are i and j respectively
//     for (int i=0; i<output_height; i++){
//         for (int j=0; j<output_width; j++){

//             for (int k=0; k<kernel_height; k++)
//                 for (int l=0; l<kernel_width; l++)
//                     for (int c=0; c<input.size(); c++)
//                         res += input[c][i+k][j+l] * kernel[c][k][l];
            
//             output[i][j] = res;
//             res = 0;
//         }
//     }

//     return output;
// }