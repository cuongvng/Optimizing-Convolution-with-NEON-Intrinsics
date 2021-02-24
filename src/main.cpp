#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "../include/conv_layer.hpp"
#include <iostream>

using namespace std;

int main()
{
  vector<vector<uint8_t>> kernel({{65,66,67}, {68,69,70}, {71,72,73}});
  uint8_t* flattened_kernel = flatten_kernel(kernel);
  for (int i=0; i<9; i++)
    cout << *(flattened_kernel+i);

}


#pragma GCC pop