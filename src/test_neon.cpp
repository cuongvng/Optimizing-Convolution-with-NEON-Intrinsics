#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <iostream>
#include <vector>
#include <chrono>
#include "../include/neon2sse/NEON_2_SSE.h"  // Replace this include by the following to test on REAL ARM machines.
// #include <arm_neon.h>

void neon_add(uint8_t* res, uint8_t* a, uint8_t* b);
void naive_add(uint8_t* res, uint8_t* a, uint8_t* b); 

enum{ARRAY_SIZE=1000000};

int main()
{   
    uint8_t a[ARRAY_SIZE];
    uint8_t b[ARRAY_SIZE];
    uint8_t res_naive[ARRAY_SIZE];
    uint8_t res_neon[ARRAY_SIZE];
    
    for (auto i=0; i<ARRAY_SIZE; i++){
        a[i] = rand();
        b[i] = rand();
    }

    naive_add(res_naive, a, b);
    neon_add(res_neon, a, b);

    // // Check equality
    if (std::equal(std::begin(res_naive), std::end(res_naive), std::begin(res_neon)))
        std::cout << "Arrays are equal.\n";
    else
        std::cout << "Arrays are NOT equal.\n";

}

void neon_add(uint8_t* res, uint8_t* a, uint8_t* b){
    auto start = std::chrono::steady_clock::now();

    for (uint32_t block16_idx=0; block16_idx<ARRAY_SIZE/16; block16_idx+=16){
        uint8x16_t block_a = vld1q_u8(a + block16_idx);
        uint8x16_t block_b = vld1q_u8(b + block16_idx);
        uint8x16_t block_res = vaddq_u8(block_a, block_b);   
        vst1q_u8(&(res[block16_idx]), block_res);
    }

    for (auto i=ARRAY_SIZE - 16*(ARRAY_SIZE/16); i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
    std::cout << "NEON addition time elapsed: " << elapsed << "(us)" << std::endl;
}

void naive_add(uint8_t* res, uint8_t* a, uint8_t* b){
    auto start = std::chrono::steady_clock::now();

    for (auto i=0; i<ARRAY_SIZE; i++)
        res[i] = a[i] + b[i];

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    
    std::cout << "Naive addition time elapsed: " << elapsed << "(us)" << std::endl;
}

#pragma GCC pop