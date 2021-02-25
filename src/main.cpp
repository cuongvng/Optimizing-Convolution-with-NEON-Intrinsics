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

  vector<vector<float32_t>> res = convolve_neon(input, kernel);
  for (int i=0; i<res.size(); i++)
    for (int j=0; j<res[0].size(); j++)
      cout << res[i][j] << '\t';
  
}


#pragma GCC pop