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

inline void display_array(const float* arr, size_t size, \
                                    std::string arrayName="") {
    if (arrayName != ""){
        std::cout << arrayName << ": ";
    }
    for (size_t i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int getError_test(){
    // input = {1,2,3,4,5}
    // target = {45}
    // hidden1 =  {h1,h2,h3}
    // output = {o1}
    // error = MSE(o1,target,size) = (target-o1)^2
    Layer x(5);
    float* myarr = new float[5];
    for (int i = 0; i < 5; i++){
        myarr[i] = i+1;
    }
    x.setNeurons(myarr, 5);
    Layer h1(x, 3);
    Layer o1(h1, 2);
    h1.setDebug(true);
    o1.setDebug(true);
    h1.forward_pass(x);
    o1.forward_pass(h1);
    // Assume all weights and bias to be 1, then after one pass - 
    // h1 = {16,16,16}
    // o1 = {49}, lets put target = {45}, then error = 4^2=16
    // display_array(h1.getNeurons(), 3, "h1");
    // display_array(o1.getNeurons(), 1, "o1");
    float* target = new float[2];
    target[0] = 45;
    target[1] = 45;
    o1.calcError(target);
    std::cout << "Error is: " << o1.getError() << std::endl;
}

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
    y.setDebug(true);
    y.forward_pass(x);
    float *result = y.getNeurons();
    for (int i = 0; i < 2; i++){
        std::cout << result[i] << "\n";
    }
    return 0;
}

int main(){
    //forward_pass_test();
    //setNeuron_test();
    getError_test();
    return 1;
}