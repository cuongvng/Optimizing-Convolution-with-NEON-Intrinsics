#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../include/convolution_neon.hpp"
#include "../include/convolution.hpp"
#include <iostream>
#include <vector>

void test1();
void test2();

int main()
{
  test1();
  test2();
}


void test1(){
  std::cout << "--------- TEST 1 -----------" << std::endl;
  // Small size input
  uint32_t input_height = 4;
  uint32_t input_width = 5;
  uint32_t kernel_height = 3;
  uint32_t kernel_width = 3;

  std::vector<std::vector<float32_t>> input_({
    {65,66,67,65,40}, 
    {68,69,70,68,40}, 
    {71,72,73,71,40}, 
    {74,75,76,74,40}});
  std::vector<std::vector<float32_t>> kernel_({
    {65,66,67}, 
    {68,69,70}, 
    {71,72,73}});

  float32_t** input = new float32_t* [input_height];
  float32_t** kernel = new float32_t* [kernel_height];

  for (auto i=0; i<input_width; i++)
    input[i] = input_[i].data();
  for (auto k=0; k<kernel_width; k++)
    kernel[k] = kernel_[k].data();

  uint32_t output_height = input_height - kernel_height + 1;
  uint32_t output_width = input_width - kernel_width + 1;

  float32_t** scalar = simply_convolve_scalar(
    input, kernel, input_height, input_width,
    kernel_height, kernel_width);

  std::cout << "Scalar output:" << std::endl;
  for (int i=0; i<output_height; i++){
    for (int j=0; j<output_width; j++)
      std::cout << scalar[i][j] << '\t';
    std::cout << std::endl;
  }

  float32_t** neon = simply_convolve_neon(
    input, kernel, input_height, input_width,
    kernel_height, kernel_width);  

  std::cout << "NEON output:" << std::endl;
  for (int i=0; i<output_height; i++){
    for (int j=0; j<output_width; j++)
      std::cout << neon[i][j] << '\t';
    std::cout << std::endl;
  }
  
  // Deallocate
  delete[] input;
  delete[] kernel;

  for (auto o=0; o<output_height; o++){
    delete[] neon[o];
    delete[] scalar[o];
  }
  delete[] neon;
  delete[] scalar;

}

void test2(){
  std::cout << std::endl << "--------- TEST 2 -----------" << std::endl;
  // Big-size input of 0s (10000 x 5000), just compare running time.
  uint32_t input_height = 10000;
  uint32_t input_width = 5000;
  uint32_t kernel_height = 11;
  uint32_t kernel_width = 9;
  uint32_t output_height = input_height - kernel_height + 1;
  uint32_t output_width = input_width - kernel_width + 1;

  float32_t** input = new float32_t* [input_height];
  float32_t** kernel = new float32_t* [kernel_height];

  for (int i=0; i<input_height; i++){
    std::vector<float32_t> r(input_width);
    input[i] = r.data();
  }
  
  for (int k=0; k<kernel_height; k++){
    std::vector<float32_t> r(kernel_width);
    kernel[k] = r.data();
  }

  float32_t** scalar = simply_convolve_scalar(
    input, kernel, input_height, input_width,
    kernel_height, kernel_width);
  float32_t** neon = simply_convolve_neon(
    input, kernel, input_height, input_width,
    kernel_height, kernel_width);  

  delete[] input;
  delete[] kernel;

  for (auto o=0; o<output_height; o++){
    delete[] neon[o];
    delete[] scalar[o];
  }
  delete[] neon;
  delete[] scalar;

}

#pragma GCC pop