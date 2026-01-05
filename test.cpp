/**
 *  Author - Satyam Gupta
 *  Date - 1/4/2026
 *  Tests
 */
#include"Layer.h"
#include"NeuralNetwork.h"
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

inline void set_array(const float* arrFrom, float* arrTo, size_t size){
    for (size_t i = 0; i < size; i++){
        arrTo[i] = arrFrom[i];
    }
}

int getError_test(){
    // input = {1,2,3,4}
    // target = {50}
    // hidden1 =  {h1,h2,h3,h4,h5}
    // output = {o1}
    // error = MSE(o1,target,size) = (target-o1)^2 = 36
    Layer x(4);
    float* myarr = new float[5];
    for (int i = 0; i < 5; i++){
        myarr[i] = i+1;
    }
    x.setNeurons(myarr, 5);
    Layer h1(x, 5, 1);
    Layer o1(h1, 1, 1);
    h1.setDebug(true);
    o1.setDebug(true);
    h1.forward_pass(x);
    o1.forward_pass(h1);
    // Assume all weights and bias to be 1, then after one pass - 
    // h1 = {16,16,16}
    // o1 = {49}, lets put target = {45}, then error = 4^2=16
    // display_array(h1.getNeurons(), 3, "h1");
    // display_array(o1.getNeurons(), 1, "o1");
    float* target = new float[1];
    target[0] = 50;
    o1.calcError(target);
    std::cout << "Error is: " << o1.getError() << std::endl;
    return 1;
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

/**
 * ============================================================
NEURAL NETWORK BACKPROPAGATION - FIRST PASS CALCULATIONS
============================================================

--- INITIAL PARAMETERS ---
Inputs (x):           [1, 2, 3, 4]
Target (y):           50
Initial Weights (w):  1.0 (all)
Initial Biases (b):   1.0 (all)
Activation Function:  ReLU
Loss Function:        MSE (Mean Squared Error)
Learning Rate (eta):  0.01

--- STEP 1: FORWARD PASS ---
1. Hidden Layer Pre-activation (zh):
   zh = (1*1 + 1*2 + 1*3 + 1*4) + 1 = 11.0
2. Hidden Layer Activation (ah):
   ah = max(0, 11) = 11.0
3. Output Layer Pre-activation (zo):
   zo = (11*1 + 11*1 + 11*1 + 11*1 + 11*1) + 1 = 56.0
4. Prediction (y_hat):
   y_hat = max(0, 56) = 56.0

--- STEP 2: ERROR/LOSS CALCULATION ---
MSE Loss = (y_hat - target)^2
Loss = (56 - 50)^2 = 6^2 = 36.0

--- STEP 3: BACKWARD PASS (GRADIENTS) ---
1. Output Gradient (dL/dy_hat): 
   2 * (56 - 50) = 12.0
2. Hidden-to-Output Weight Gradient (dL/d_who):
   12.0 * 1.0 (ReLU') * 11.0 (ah) = 132.0
3. Output Bias Gradient (dL/d_bo):
   12.0 * 1.0 = 12.0
4. Input-to-Hidden Weight Gradients (dL/d_wxh):
   Formula: dL/dzo * who * ReLU' * x_i
   Grad w_x1 (x=1): 12 * 1 * 1 * 1 = 12.0
   Grad w_x2 (x=2): 12 * 1 * 1 * 2 = 24.0
   Grad w_x3 (x=3): 12 * 1 * 1 * 3 = 36.0
   Grad w_x4 (x=4): 12 * 1 * 1 * 4 = 48.0
5. Hidden Bias Gradient (dL/d_bh):
   12 * 1 * 1 = 12.0

--- STEP 4: UPDATE STEP (NEW VALUES) ---
(Rule: New = Old - eta * Gradient)

1. New Output Bias (bo):
   1.0 - (0.01 * 12.0) = 0.88
2. New Hidden-to-Output Weights (who):
   1.0 - (0.01 * 132.0) = -0.32
3. New Hidden Biases (bh):
   1.0 - (0.01 * 12.0) = 0.88
4. New Input-to-Hidden Weights (wxh):
   w_x1: 1.0 - 0.12 = 0.88
   w_x2: 1.0 - 0.24 = 0.76
   w_x3: 1.0 - 0.36 = 0.64
   w_x4: 1.0 - 0.48 = 0.52

============================================================
FINAL SUMMARY (AFTER 1ST PASS)
============================================================
Error:        36.0
New Output w: -0.32
New Output b: 0.88
New Hidden b: 0.88
New Input w:  [0.88, 0.76, 0.64, 0.52]
============================================================
*/
int backward_pass_test(){
    Layer x(4);
    float input_layer[] = {1,2,3,4};
    set_array(input_layer, x.getNeurons(), 4);
    Layer h1(x,5,1.0f);
    Layer output(h1,1,1.0f);
    h1.setDebug(true);
    output.setDebug(true);
    h1.forward_pass(x);
    output.forward_pass(h1);
    float learning_rate = 0.01;
    // For output layer assume delta_layer = derivative_MSE
    float delta_layer[] = {2.0f*(56.0f-50.0f)};
    output.backward_pass(delta_layer, h1, learning_rate);
    h1.backward_pass(output, x, learning_rate);
    return 0;
}

int read_input_test(){
    NeuralNetwork nn;
    nn.read_input("mnist_train.csv");
    nn.display_input(5);
    return 1;
}

int neural_network_structure_test(){
    NeuralNetwork nn;
    nn.add_layer(5);
    nn.add_layer(36);
    nn.add_layer(47);
    nn.add_layer(4);
    nn.display_layers();
    return 1;
}

int main(){
    //forward_pass_test();
    //setNeuron_test();
    //getError_test();
    //backward_pass_test();
    // read_input_test();
    neural_network_structure_test();
    return 1;
}