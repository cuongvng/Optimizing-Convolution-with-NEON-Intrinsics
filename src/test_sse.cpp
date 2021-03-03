// #include <xmmintrin.h>     //SSE
// #include <emmintrin.h>     //SSE2
// #include <pmmintrin.h>     //SSE3
// #include <smmintrin.h>     //SSE4.1
#include <nmmintrin.h> //SSE4.2

#include <iostream>
#include <chrono>

void sse_add(float* res, float* a, float* b);
void naive_add(float* res, float* a, float* b); 
void print128_num(__m128 num)
{
    float val[4];
    memcpy(val, &num, sizeof(val));
    printf("Numerical: %2f %2f %2f %2f \n", 
           val[0], val[1], val[2], val[3]);
}

enum{
    ARRAY_SIZE=100000,
    BLOCK_SIZE=4,
    N_CALLS = 10000
};

int main(){

    float a1[4] = { 1.0, 2.0, 3.0, 4.0 };
    float a2[4] = { 1.0, 1.0, 1.0, 1.0 };
    float s[4] = {0,0,0,0};
    /*** SCALAR ***/
    auto start = std::chrono::steady_clock::now();
    
    for (auto it=0; it<N_CALLS; it++){
        for (auto i=0; i<4; i++)
            s[i] = a1[i] + a2[i];
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
    std::cout << "Scalar: " << elapsed << "(us)"
              << " for " << N_CALLS << " calls." << std::endl;

    /*** SSE4.2 ***/
    auto start2 = std::chrono::steady_clock::now();


    for (auto it=0; it<N_CALLS; it++){
        // __m128 vector1 = _mm_set_ps(4.0, 3.0, 2.0, 1.0); // high element first, opposite of C array order.  Use _mm_setr_ps if you want "little endian" element order in the source.
        // __m128 vector2 = _mm_set_ps(1.0, 1.0, 1.0, 1.0);
        __m128 vector1 = {4.0, 3.0, 2.0, 1.0};
        __m128 vector2 = {1.0, 1.0, 1.0, 1.0};
        __m128 sum = _mm_add_ps(vector1, vector2); // result = vector1 + vector 2
    }
    auto end2 = std::chrono::steady_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count();
    
    std::cout << "SSE4.2: " << elapsed2 << "(us)" 
              << " for " << N_CALLS << " calls." << std::endl;

    return 0;
}
