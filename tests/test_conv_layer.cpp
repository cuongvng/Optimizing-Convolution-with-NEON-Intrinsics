#include "../conv/conv_layer.hpp"
#include <iostream>

using namespace std;

int main(){
  vector<vector<double>> input({{0,1,2}, {3,4,5}, {6,7,8}});
  vector<vector<double>> kernel({{0,1}, {2,3}});
  vector<vector<double>>& output = convolute(input, kernel);

  for (int i=0; i<output.size(); i++)
    for (int j=0; j<output[0].size(); j++)
      std::cout << output[i][j] << "\t";

  return 0;
}