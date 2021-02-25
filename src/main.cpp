#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../include/convolution.hpp"
#include <iostream>

using namespace std;

int main()
{
  vector<vector<float32_t>> input({{65,66,67, 65}, {68,69,70, 68}, {71,72,73, 71}, {74,75,76,74}});
  vector<vector<float32_t>> kernel({{65,66,67}, {68,69,70}, {71,72,73}});

  float32_t** res = convolve_neon(input, kernel);
  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++)
      cout << res[i][j] << '\t';
  
  // delete res
  for (int i=0; i< 2; i++)
    delete[] res[i];
  delete[] res;
}


#pragma GCC pop