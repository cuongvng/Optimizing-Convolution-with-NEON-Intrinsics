#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../include/convolution_neon.hpp"
#include "../include/convolution.hpp"
#include <iostream>

using namespace std;

int main()
{
  vector<vector<float32_t>> input({{65,66,67, 65, 40}, {68,69,70, 68, 40}, {71,72,73, 71, 40}, {74,75,76,74, 40}});
  vector<vector<float32_t>> kernel({{65,66,67}, {68,69,70}, {71,72,73}});

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


#pragma GCC pop