#include <iostream>
#include <cstdlib>

using namespace std;

// Regular convolution function.
void convolute(int ** output, int ** input, int ** kernel)
{
    int convolute = 0; // This holds the convolution results for an index.
    int x, y; // Used for input matrix index

	// Fill output matrix: rows and columns are i and j respectively
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			x = i;
			y = j;

			// Kernel rows and columns are k and l respectively
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					// Convolute here.
					convolute += kernel[k][l] * input[x][y];
					y++; // Move right.
				}
				x++; // Move down.
				y = j; // Restart column position
			}
			output[i][j] = convolute; // Add result to output matrix.
			convolute = 0; // Needed before we move on to the next index.
		}
	}
}

int main(int argc, char * argv[])
{

    int ** kernel = new int*[3];

    for(int i = 0; i < 3; ++i)
    {
        kernel[i] = new int[3];
    }

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
            kernel[i][j] = 1;
    }

    int ** matrixIn = new int*[8];
    for(int i = 0; i < 8; ++i)
    {
        matrixIn[i] = new int[8];
    }

    for(int i = 0; i < 8; i++)
    {
        for(int j = 0; j < 8; j++)
            matrixIn[i][j] = 1;
    }

    int ** output = new int*[6];
    for(int i = 0; i < 6; ++i)
    {
        output[i] = new int[6];
    }

    for(int i = 0; i < 6; i++)
        output[i][i] = 0;

    convolute(output, matrixIn, kernel);

    // Print out the resulting matrix.
    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            cout << output[i][j] << " ";
        }
        cout << endl;
    }

    delete[] kernel;
    delete[] matrixIn;
    delete[] output;

    return 0;
}