#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../include/convolution_neon.hpp"
#include "../include/convolution.hpp"
#include <iostream>
#include <vector>
#include <chrono>
enum{
    N_CALLS = 10000
};

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
  uint32_t input_width = 4;
  uint32_t kernel_height = 2;
  uint32_t kernel_width = 2;

  std::vector<std::vector<float32_t>> input_({
    {65,66,67,65}, 
    {68,69,70,68}, 
    {71,72,73,71}, 
    {74,75,76,74}});
  std::vector<std::vector<float32_t>> kernel_({
    {65,66}, 
    {68,69}}
    );

  float32_t** input = new float32_t* [input_height];
  float32_t** kernel = new float32_t* [kernel_height];

  for (auto i=0; i<input_width; i++)
    input[i] = input_[i].data();
  for (auto k=0; k<kernel_width; k++)
    kernel[k] = kernel_[k].data();

  uint32_t output_height = input_height - kernel_height + 1;
  uint32_t output_width = input_width - kernel_width + 1;
  
  float32_t** scalar = new float32_t* [output_height];
  float32_t** neon = new float32_t* [output_height];
  for (auto o=0; o<output_height; o++){
    scalar[o] = new float32_t [output_width];
    neon[o] = new float32_t [output_width];
  }

  auto start = std::chrono::steady_clock::now();
  for (auto it=0; it<N_CALLS; it++){
    simply_convolve_scalar(
      scalar, input, kernel, input_height, input_width,
      kernel_height, kernel_width);
  }
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  
  std::cout << "Scalar: " << elapsed << "(us)"
            << " for " << N_CALLS << " calls." << std::endl;
  std::cout << "Scalar output:" << std::endl;
  for (int i=0; i<output_height; i++){
    for (int j=0; j<output_width; j++)
      std::cout << scalar[i][j] << '\t';
    std::cout << std::endl;
  }

  auto start2 = std::chrono::steady_clock::now();

  for (auto it=0; it<N_CALLS; it++){
    simply_convolve_neon(
      neon, input, kernel, input_height, input_width,
      kernel_height, kernel_width);  
  }
  auto end2 = std::chrono::steady_clock::now();
  auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count();
  
  std::cout << "NEON: " << elapsed2 << "(us)" 
            << " for " << N_CALLS << " calls." << std::endl;

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
  float32_t** scalar = new float32_t* [output_height];
  float32_t** neon = new float32_t* [output_height];

  for (int i=0; i<input_height; i++){
    std::vector<float32_t> r(input_width);
    input[i] = r.data();
  }
  
  for (int k=0; k<kernel_height; k++){
    std::vector<float32_t> r(kernel_width);
    kernel[k] = r.data();
  }

  for (int o=0; o<output_height; o++){
    scalar[o] = new float32_t [output_width];
    neon[o] = new float32_t [output_width];
  }

  auto start = std::chrono::steady_clock::now();
  simply_convolve_scalar(
    scalar, input, kernel, input_height, input_width,
    kernel_height, kernel_width);
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  
  std::cout << "Scalar: " << elapsed << "(us)" << std::endl;

  auto start2 = std::chrono::steady_clock::now();
  simply_convolve_neon(
    neon, input, kernel, input_height, input_width,
    kernel_height, kernel_width);  
  auto end2 = std::chrono::steady_clock::now();
  auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count();
  
  std::cout << "NEON: " << elapsed2 << "(us)" << std::endl;

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