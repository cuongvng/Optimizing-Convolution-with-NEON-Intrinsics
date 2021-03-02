#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <iostream>
#include <vector>
#include <chrono>
#include "../include/neon2sse/NEON_2_SSE.h"  // Replace this include by the following to test on REAL ARM machines.
// #include <arm_neon.h>

void neon_add(float32_t* res, float32_t* a, float32_t* b);
void naive_add(float32_t* res, float32_t* a, float32_t* b); 

enum{
    ARRAY_SIZE=100000,
    BLOCK_SIZE=4,
    N_CALLS = 10000
};

int main(){

    float32_t a1[4] = { 1.0, 2.0, 3.0, 4.0 };
    float32_t a2[4] = { 1.0, 1.0, 1.0, 1.0 };
    float32_t s[4] = {0,0,0,0};
    /*** NAIVE ***/
    auto start = std::chrono::steady_clock::now();
    
    for (auto it=0; it<N_CALLS; it++){
        for (auto i=0; i<4; i++)
            s[i] = a1[i] + a2[i];
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
    std::cout << "Naive addition time elapsed: " << elapsed << "(us)"
              << " for " << N_CALLS << " calls." << std::endl;

    /*** NEON ***/
    float32_t v1[4] = { 1.0, 2.0, 3.0, 4.0 };
    float32_t v2[4] = { 1.0, 1.0, 1.0, 1.0 };
    float32_t z[4];
    
    auto start2 = std::chrono::steady_clock::now();

    for (auto it=0; it<N_CALLS; it++){
        float32x4_t v1_ = vld1q_f32(v1);
        float32x4_t v2_ = vld1q_f32(v2);
        float32x4_t sum = vaddq_f32(v1_, v2_);
        vst1q_f32(z, sum);
    }
    auto end2 = std::chrono::steady_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count();
    
    std::cout << "NEON addition time elapsed: " << elapsed2 << "(us)" 
              << " for " << N_CALLS << " calls." << std::endl;

    return 0;
}

// int main()
// {   
//     float32_t a[ARRAY_SIZE];
//     float32_t b[ARRAY_SIZE];
//     float32_t res_naive[ARRAY_SIZE];
//     float32_t res_neon[ARRAY_SIZE];
    
//     for (auto i=0; i<ARRAY_SIZE; i++){
//         a[i] = rand()/float(RAND_MAX);
//         b[i] = rand()/float(RAND_MAX);
//     }

//     /*** NAIVE ***/
//     auto start = std::chrono::steady_clock::now();
    
//     for (auto it=0; it<N_CALLS; it++)
//         naive_add(res_naive, a, b);
    
//     auto end = std::chrono::steady_clock::now();
//     auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
//     std::cout << "Naive addition time elapsed: " << elapsed << "(us)"
//               << " for " << N_CALLS << " calls." << std::endl;

//     /*** NEON ***/
//     auto start2 = std::chrono::steady_clock::now();

//     for (auto it=0; it<N_CALLS; it++)
//         neon_add(res_neon, a, b);
    
//     auto end2 = std::chrono::steady_clock::now();
//     auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count();
    
//     std::cout << "NEON addition time elapsed: " << elapsed2 << "(us)" 
//               << " for " << N_CALLS << " calls." << std::endl;

//     // // Check equality
//     if (std::equal(std::begin(res_naive), std::end(res_naive), std::begin(res_neon)))
//         std::cout << "Arrays are equal.\n";
//     else
//         std::cout << "Arrays are NOT equal.\n";

// }

void neon_add(float32_t* res, float32_t* a, float32_t* b){
    for (uint32_t blockidx=0; blockidx<ARRAY_SIZE/BLOCK_SIZE; blockidx+=BLOCK_SIZE){
        float32x4_t block_a = vld1q_f32(a + blockidx);
        float32x4_t block_b = vld1q_f32(b + blockidx);
        float32x4_t block_res = vaddq_f32(block_a, block_b);   
        vst1q_f32(&(res[blockidx]), block_res);
    }

    for (auto i=ARRAY_SIZE - BLOCK_SIZE*(ARRAY_SIZE/BLOCK_SIZE); i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];
}

void naive_add(float32_t* res, float32_t* a, float32_t* b){
    for (auto i=0; i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];
}

#pragma GCC pop