#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <iostream>
#include <vector>
#include <chrono>
#include "../include/neon2sse/NEON_2_SSE.h"  // Replace this include by the following to test on REAL ARM machines.
// #include <arm_neon.h>


void neon_add(float32_t* res, float32_t* a, float32_t* b);
void naive_add(float32_t* res, float32_t* a, float32_t* b); 

enum{ARRAY_SIZE=10000};

int main()
{   
    float32_t a[ARRAY_SIZE];
    float32_t b[ARRAY_SIZE];
    float32_t res_naive[ARRAY_SIZE];
    float32_t res_neon[ARRAY_SIZE];
    
    for (auto i=0; i<ARRAY_SIZE; i++){
        a[i] = rand()/float(RAND_MAX);
        b[i] = rand()/float(RAND_MAX);
    }

    naive_add(res_naive, a, b);
    neon_add(res_neon, a, b);

    // // Check equality
    if (std::equal(std::begin(res_naive), std::end(res_naive), std::begin(res_neon)))
        std::cout << "Arrays are equal.\n";
    else
        std::cout << "Arrays are NOT equal.\n";

}

void neon_add(float32_t* res, float32_t* a, float32_t* b){
    auto start = std::chrono::steady_clock::now();

    for (uint32_t block4_idx=0; block4_idx<ARRAY_SIZE/4; block4_idx+=4){
        float32x4_t block_a = vld1q_f32(a + block4_idx);
        float32x4_t block_b = vld1q_f32(b + block4_idx);
        float32x4_t block_res = vaddq_f32(block_a, block_b);   
        vst1q_f32(&(res[block4_idx]), block_res);
    }

    for (auto i=ARRAY_SIZE - 4*(ARRAY_SIZE/4); i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
    std::cout << "NEON addition time elapsed: " << elapsed << "(us)" << std::endl;
}

void naive_add(float32_t* res, float32_t* a, float32_t* b){
    auto start = std::chrono::steady_clock::now();

    for (auto i=0; i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
    std::cout << "Naive addition time elapsed: " << elapsed << "(us)" << std::endl;
}

#pragma GCC pop