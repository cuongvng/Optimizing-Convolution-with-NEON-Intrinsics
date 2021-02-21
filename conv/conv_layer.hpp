#include<vector>

using namespace std;

// 1D convolution function.
vector<vector<double>>& convolute(vector<vector<double>> input, vector<vector<double>> kernel)
{   
    int input_height = input.size();
    int input_width = input[0].size();
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    static vector<vector<double>> output;
    for (int i=0; i<output_height; i++)
        output.push_back(vector<double> (output_width, 0.0));

    double convolute = 0; // This holds the convolution results for an index.

	// Fill output matrix: rows and columns are i and j respectively
    for (int i=0; i<output_height; i++){
        for (int j=0; j<output_width; j++){

            for (int k=0; k<kernel_height; k++){
                for (int l=0; l<kernel_width; l++){
                    convolute += input[i+k][j+l] * kernel[k][l];
                }
            }
            output[i][j] = convolute;
            convolute = 0;
        }
    }

    return output;
}