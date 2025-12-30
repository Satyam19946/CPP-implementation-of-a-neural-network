/**
 * Author - Satyam Gupta
 * Date - 12/28/2025
 * Definitions of various cost functions
 */

#include<cstring>
#pragma once

namespace CostFuncs {
    float MSE (float* y, float* target, size_t size){
        float total = 0.0f;
        for (size_t i = 0; i < size; i++){
            // Direct multiplication is faster than std::powf or std::expf
            total += (y[i]-target[i])*(y[i]-target[i]);
        }
        return total;
    };
}