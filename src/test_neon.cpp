#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <iostream>
#include <vector>
#include <chrono>
#include "../include/neon2sse/NEON_2_SSE.h"  // Replace this include by the following to test on REAL ARM machines.
// #include <arm_neon.h>

void neon_add(uint16_t* res, uint16_t* a, uint16_t* b);
void naive_add(uint16_t* res, uint16_t* a, uint16_t* b); 

enum{
    ARRAY_SIZE=100000,
    BLOCK_SIZE=16,
    N_CALLS = 10000
};

int main()
{   
    uint16_t a[ARRAY_SIZE];
    uint16_t b[ARRAY_SIZE];
    uint16_t res_naive[ARRAY_SIZE];
    uint16_t res_neon[ARRAY_SIZE];
    
    for (auto i=0; i<ARRAY_SIZE; i++){
        a[i] = rand();
        b[i] = rand();
    }

    /*** NAIVE ***/
    auto start = std::chrono::steady_clock::now();
    
    for (auto it=0; it<N_CALLS; it++)
        naive_add(res_naive, a, b);
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
    std::cout << "Naive addition time elapsed: " << elapsed << "(us)"
              << " for " << N_CALLS << " calls." << std::endl;

    /*** NEON ***/
    auto start2 = std::chrono::steady_clock::now();

    for (auto it=0; it<N_CALLS; it++)
        neon_add(res_neon, a, b);
    
    auto end2 = std::chrono::steady_clock::now();
    auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count();
    
    std::cout << "NEON addition time elapsed: " << elapsed2 << "(us)" 
              << " for " << N_CALLS << " calls." << std::endl;

    // // Check equality
    if (std::equal(std::begin(res_naive), std::end(res_naive), std::begin(res_neon)))
        std::cout << "Arrays are equal.\n";
    else
        std::cout << "Arrays are NOT equal.\n";

}

void neon_add(uint16_t* res, uint16_t* a, uint16_t* b){
    for (uint32_t blockidx=0; blockidx<ARRAY_SIZE/BLOCK_SIZE; blockidx+=BLOCK_SIZE){
        uint8x16_t block_a = vld1q_u8(a + blockidx);
        uint8x16_t block_b = vld1q_u8(b + blockidx);
        uint8x16_t block_res = vaddq_u8(block_a, block_b);   
        vst1q_u8(&(res[blockidx]), block_res);
    }

    for (auto i=ARRAY_SIZE - BLOCK_SIZE*(ARRAY_SIZE/BLOCK_SIZE); i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];
}

void naive_add(uint16_t* res, uint16_t* a, uint16_t* b){
    for (auto i=0; i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];
}

#pragma GCC pop