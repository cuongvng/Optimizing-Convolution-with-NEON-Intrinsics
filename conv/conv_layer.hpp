#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include <armadillo>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#define DEBUG false
#define DEBUG_PREFIX

class Conv2d{
    private:
        size_t in_height;
        size_t inWidth;
        size_t inChannels;
        size_t outChannels;
        size_t kernelHeight;
        size_t kernelWidth;
        size_t horizontalStride;
        size_t verticalStride;

        std::vector<arma::cube> kernels;
        arma::cube input;
        arma::cube output;
        arma::cube gradInput;
        arma::cube accumulatedGradInput;
        std::vector<arma::cube> gradKernels;
        std::vector<arma::cube> accumulatedGradKernels;

    public:
        Conv2d(size_t inHeight, size_t inWidth, size_t inChannels, 
            size_t kernelHeight, size_t kernelWidth, size_t outChannels,
            size_t horizontalStride, size_t verticalStride):{
            // Initialize kernels
            for(int k=0; k<outChannels; k++)
                this->kernels.push_back(arma::zeros(kernelHeight, kernelWidth));

            // Clean up gradient
            this->_zero_grad();
        }

        arma::cube& forward(arma::cube& input){
            assert((in_height - kernelHeight) % verticalStride == 0);
            assert((inWidth - kernelWidth) % horizontalStride == 0);

            // Initialize output
            output = arma::zeros(
                n_rows = (inHeight - kernelHeight)/verticalStride + 1,
                n_cols = (inWidth - kernelWidth)/horizontalStride + 1,
                n_slices = outChannels
            );

            // Perform the convolutional operation
            for(int k=0; k<outChannels; k++){
                for (int i=0; i<inHeight-kernelHeight+1; i+=verticalStride)
                    for (int j=0; j<inWidth-kernelWidth+1; j+=horizontalStride)
                        output(i/verticalStride, j/horizontalStride, k) = arma::dot(
                            arma::vectorise(input.subcube(i, j, 0, i+kernelHeight-1, j+kernelWidth-1, outChannels-1))),
                            arma::vectorise(this->kernels[k]);
            }
            this->input = input;
            this->output = output;
            return output;
        }

        void backward(arma::cube& gradient){
            
        }
    
    private:
        void _zero_grad(){
            
        }
}

#undef DEBUG
#undef DEBUG_PREFIX