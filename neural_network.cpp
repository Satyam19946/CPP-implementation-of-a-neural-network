/*  
    Author - Satyam Gupta 
    Date - 12/25/25
    A neural network consists of the following - 
        1. An input layer - reads the data (x)
        2. n number of hidden layers (performs computations on x)
        3. An output layer - used for predictions (y) 
*/
#include"Layer.h"
#include<iostream>

int setNeuron_test(){
    size_t size = 100;
    Layer x(100);
    float* myarr = new float[size];
    for (int i = 0; i < size; i++){
        myarr[i] = i*size;
    }
    x.setNeurons(myarr, size);
    float* X_neurons = x.getNeurons();
    std::cout << "X is: \n";
    std::cout <<sizeof(X_neurons) << " " <<sizeof(myarr) << "\n";
    for (int i = 0; i < size; i++){
        std::cout<<X_neurons[i] << "\n";
    }
    return 1;
}

int forward_pass_test(){
    Layer x(4);
    float myarr[4] = {1,2,3,4};
    x.setNeurons(myarr, 4);
    Layer y(x, 2);
    y.forward_pass(x);
    float *result = y.getNeurons();
    for (int i = 0; i < 2; i++){
        std::cout << result[i] << "\n";
    }
    return 0;
}

int main(){
    forward_pass_test();
    return 1;
}