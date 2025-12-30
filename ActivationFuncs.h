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

    constexpr float derivate_ReLU(float x){
        if (x < 0){
            return 0.0f;
        } else {
            return 1.0f;
        }
    };

    //todo - Sigmoid, tanh, softmax
}