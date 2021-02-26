#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../include/convolution_neon.hpp"
#include "../include/convolution.hpp"
#include <iostream>

using namespace std;

void test1(){
  cout << "--------- TEST 1 -----------" << endl;
  // Small size input
  vector<vector<float32_t>> input({
    {65,66,67,65,40}, 
    {68,69,70,68,40}, 
    {71,72,73, 71,40}, 
    {74,75,76,74,40}});
  vector<vector<float32_t>> kernel({
    {65,66,67}, 
    {68,69,70}, 
    {71,72,73}});

  cout << "None-NEON results:" << endl;
  vector<vector<float32_t>> non_neon = simply_convolve(input, kernel);
  for (int i=0; i<non_neon.size(); i++){
    for (int j=0; j<non_neon[0].size(); j++)
      cout << non_neon[i][j] << '\t';
    cout << endl;
  }

  cout << "NEON results:" << endl;
  vector<vector<float32_t>> neon = simply_convolve_neon(input, kernel);
  for (int i=0; i<neon.size(); i++){
    for (int j=0; j<neon[0].size(); j++)
      cout << neon[i][j] << '\t';
    cout << endl;
  }

}

void test2(){
  cout << endl << "--------- TEST 2 -----------" << endl;
  // Big-size input of 0s (10000 x 5000), just compare running time.
  vector<vector<float32_t>> input{}; // 10000 x 5000
  vector<vector<float32_t>> kernel{}; // 11x9

  for (int i=0; i<10000; i++)
    input.push_back(vector<float32_t> (5000));
  
  for (int i=0; i<11; i++)
    kernel.push_back(vector<float32_t> (9));

  vector<vector<float32_t>> non_neon = simply_convolve(input, kernel);
  vector<vector<float32_t>> neon = simply_convolve_neon(input, kernel);
}

int main()
{
  test1();
  test2();
}


#pragma GCC pop