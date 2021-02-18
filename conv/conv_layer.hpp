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
            size_t horizontalStride, size_t verticalStride
        ):{
            // Initialize kernels
            for(int k=0; k<outChannels; k++)
                self->kernels.push_back(arma::zeros(kernelHeight, kernelWidth))

            // Clean up gradient
            self._zero_grad()
        }

        arma::cube& forward(arma::cube& input){
        }

        void backward(arma::cube& gradient){

        }
    
    private:
        void _zero_grad(){
            
        }
}

#undef DEBUG
#undef DEBUG_PREFIX