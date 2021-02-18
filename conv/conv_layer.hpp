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
            for (size_t k=0; k<this->out_channels; k++)
                this->kernels.push_back(arma::zeros(kernel_height, kernel_width, in_channels));

            // Clean up gradient
            this->_reset_accumulated_grad();
        }

        arma::cube& forward(arma::cube& input){
            assert((this->in_height - this->kernel_height) % this->vertical_stride == 0);
            assert((this->in_width - this->kernel_width) % this->horiz_stride == 0);

            // Initialize output
            output = arma::zeros(
                n_rows = (this->in_height - this->kernel_height)/this->vertical_stride + 1,
                n_cols = (this->in_width - this->kernel_width)/this->horiz_stride + 1,
                n_slices = this->out_channels
            );

            // Perform the convolutional operation
            for (size_t k=0; k<this->out_channels; k++){
                for (size_t i=0; i<this->in_height-this->kernel_height+1; i+=this->vertical_stride)
                    for (size_t j=0; j<this->in_width-this->kernel_width+1; j+=this->horiz_stride)
                        output(i/this->vertical_stride, j/this->horiz_stride, k) = arma::dot(
                            arma::vectorise(input.subcube(i, j, 0, i+this->kernel_height-1, j+this->kernel_width-1, this->in_channels-1))),
                            arma::vectorise(this->kernels[k]);
            }
            this->input = input;
            return output;
        }

        void backward(arma::cube& upstream_gradient, arma::cube& output){
            // Initialize input gradient
            this->grad_input = arma::zeros(arma::size(input))

            // Compute gradient wrt input
            for (size_t k=0; k<this->out_channels; k++)
                for (size_t i=0; i<output.n_rows; i++)
                    for (size_t j=0; j<output.n_cols; j++){
                        arma::cube tmp(arma::size(input), arma::fill:zeros);
                        tmp.subcube(i*this->vertical_stride, 
                                    j*this->horiz_stride, 
                                    0,
                                    i*this->vertical_stride + this->kernel_height-1, 
                                    j*this->horiz_stride + this->kernel_width-1, 
                                    this->in_channels-1)
                        = this->kernels[k];

                        this->grad_input += upstream_gradient.slice(k)(i, j) * tmp;
                    }
            
            this->accumulated_grad_input += this->grad_input;

            // Initialize gradient wrt filters
            this->grad_kernels.clear();
            this->grad_kernels.resize(this->out_channels);
            for (size_t k=0; k<this->out_channels; k++)
                this->grad_kernels[k] = arma::zeros(this->kernel_height, this->kernel_width, this->in_channels);
            
            // Compute gradient wrt filters
            for (size_t k=0; k<this->out_channels; k++)
                for (size_t i=0; i<output.n_rows; i++)
                    for (size_t j=0; j<output.n_cols; j++){
                        arma::cube tmp(arma::size(kernels[k], arma::fill:zeros));
                        tmp = this->input.subcube(i*this->vertical_stride, 
                                                  j*this->horiz_stride, 
                                                  0,
                                                  i*this->vertical_stride + this->kernel_height-1, 
                                                  j*this->horiz_stride + this->kernel_width-1,
                                                  this->in_channels-1);
                        
                        this->grad_kernels[k] += upstream_gradient.slice(k)(i, j) * tmp;
                    }
            
            for (size_t k=0; k<this->out_channels; k++)
                this->accumulated_grad_kernels[k] += this->grad_kernels[k]

        }

        void update_kernels(size_t batch_size, double lr){
            for (size_t k=0; k<this->out_channels; k++)
                this->kernels[k] -= lr * (this->accumulated_grad_kernels[k]/batch_size);

            this->_reset_accumulated_grad();
        }

        void set_kernels(std::vector<arma::cube> kernels){
            this->kernels = kernels;
        }

        std::vector<arma::cube> get_kernels(){
            return this->kernels;
        }

        arma::cube get_grad_wrt_input(){
            return this->grad_input;
        }

        std::vector<arma::cube> get_grad_wrt_kernels(){
            return this->grad_kernels
        }

    private:
        void _reset_accumulated_grad(){
            // Reset grad wrt input
            this->accumulated_grad_input = arma::zeros(this->in_height,
                                                       this->in_width,
                                                       this->in_channels);
            
            // Reset grad wrt kernels
            this->accumulated_grad_kernels.clear();
            this->accumulated_grad_kernels.resize(this->out_channels);
            for (size_t k=0; k<this->out_channels; k++)
                this->accumulated_grad_kernels[k] = arma::zeros(this->kernel_height,
                                                                this->kernel_width,
                                                                this->in_channels);
        }
}

#undef DEBUG
#undef DEBUG_PREFIX