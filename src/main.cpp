#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../include/convolution_neon.hpp"
#include "../include/convolution.hpp"
#include <iostream>
#include <vector>

void test1(){
  std::cout << "--------- TEST 1 -----------" << std::endl;
  // Small size input
  std::vector<std::vector<float32_t>> input({
    {65,66,67,65,40}, 
    {68,69,70,68,40}, 
    {71,72,73, 71,40}, 
    {74,75,76,74,40}});
  std::vector<std::vector<float32_t>> kernel({
    {65,66,67}, 
    {68,69,70}, 
    {71,72,73}});

  std::vector<std::vector<float32_t>> non_neon = simply_convolve(input, kernel);
  std::cout << "None-NEON output:" << std::endl;
  for (int i=0; i<non_neon.size(); i++){
    for (int j=0; j<non_neon[0].size(); j++)
      std::cout << non_neon[i][j] << '\t';
    std::cout << std::endl;
  }

  std::vector<std::vector<float32_t>> neon = simply_convolve_neon(input, kernel);
  std::cout << "NEON output:" << std::endl;
  for (int i=0; i<neon.size(); i++){
    for (int j=0; j<neon[0].size(); j++)
      std::cout << neon[i][j] << '\t';
    std::cout << std::endl;
  }

}

void test2(){
  std::cout << std::endl << "--------- TEST 2 -----------" << std::endl;
  // Big-size input of 0s (10000 x 5000), just compare running time.
  std::vector<std::vector<float32_t>> input{}; // 10000 x 5000
  std::vector<std::vector<float32_t>> kernel{}; // 11x9

  for (int i=0; i<10000; i++)
    input.push_back(std::vector<float32_t> (5000));
  
  for (int i=0; i<11; i++)
    kernel.push_back(std::vector<float32_t> (9));

  std::vector<std::vector<float32_t>> non_neon = simply_convolve(input, kernel);
  std::vector<std::vector<float32_t>> neon = simply_convolve_neon(input, kernel);
}

int main()
{
  test1();
  test2();
}


#pragma GCC pop