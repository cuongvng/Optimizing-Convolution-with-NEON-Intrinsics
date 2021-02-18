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
        size_t in_width;
        size_t in_channels;
        size_t out_channels;
        size_t kernel_height;
        size_t kernel_width;
        size_t horiz_stride;
        size_t vertical_stride;

        std::vector<arma::cube> kernels;
        arma::cube input;
        arma::cube output;
        arma::cube grad_input;
        arma::cube accumulated_grad_input;
        std::vector<arma::cube> grad_kernels;
        std::vector<arma::cube> accumulated_grad_kernels;

    public:
        Conv2d(size_t in_height, size_t in_width, size_t in_channels, 
            size_t kernel_height, size_t kernel_width, size_t out_channels,
            size_t horiz_stride, size_t vertical_stride):{
            // Initialize kernels
            for(size_t k=0; k<out_channels; k++)
                this->kernels.push_back(arma::zeros(kernel_height, kernel_width));

            // Clean up gradient
            this->_zero_grad();
        }

        arma::cube& forward(arma::cube& input){
            assert((in_height - kernel_height) % vertical_stride == 0);
            assert((in_width - kernel_width) % horiz_stride == 0);

            // Initialize output
            output = arma::zeros(
                n_rows = (in_height - kernel_height)/vertical_stride + 1,
                n_cols = (in_width - kernel_width)/horiz_stride + 1,
                n_slices = out_channels
            );

            // Perform the convolutional operation
            for(size_t k=0; k<out_channels; k++){
                for (size_t i=0; i<in_height-kernel_height+1; i+=vertical_stride)
                    for (size_t j=0; j<in_width-kernel_width+1; j+=horiz_stride)
                        output(i/vertical_stride, j/horiz_stride, k) = arma::dot(
                            arma::vectorise(input.subcube(i, j, 0, i+kernel_height-1, j+kernel_width-1, out_channels-1))),
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