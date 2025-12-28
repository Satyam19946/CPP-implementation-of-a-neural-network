/**
 *  Author - Satyam Gupta
 *  Date - 12/27/2025
 *  Contains the definitions of all activation functions
 */


#pragma once

namespace ActivationFuncs {
    constexpr float ReLU(float x) {
        if (x < 0){
            return 0.0f;
        } else {
            return x;
        }
    };

    constexpr float DeltaReLU(float x){
        if (x < 0){
            return 0.0f;
        } else {
            return x;
        }
    };

    //todo - Sigmoid, tanh, softmax
}