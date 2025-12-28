/*
    Author - Satyam Gupta
    Date - 12/25/25
    Implementation of the Layer class defined in Layer.h
*/

#include"Layer.h"
#include"ActivationFuncs.h"
#include"CostFuncs.h"
#include<cstring>
#include<random>
#include<iostream>

size_t Layer::number_layers;
bool Layer::debug = false;

Layer::Layer(size_t n_neurons){
    this->n_neurons = n_neurons;
    neurons = new float[n_neurons];
    number_layers += 1;
}

Layer::Layer(const Layer& prev, size_t n_neurons, float init_value){
    input_n_neurons = prev.getSize();
    weights = new float[n_neurons * input_n_neurons]; \
                // input=4x1, output=6x1 => weights=6x4
    bias = new float[n_neurons];
    neurons = new float[n_neurons];
    this->n_neurons = n_neurons;
    number_layers += 1;
    this->initialize_weights(init_value);
}

void Layer::initialize_weights(float init_value) {
    // initialize the weights and bias
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(0.0f, 1.0f); 
    for (size_t i = 0; i < n_neurons; i++){
        for (size_t j=0; j < input_n_neurons; j++){
            if (init_value == 0.0f) {
                weights[i*input_n_neurons + j] = distr(gen);
                bias[i] = distr(gen);
            } else {
                weights[i*input_n_neurons + j] = init_value;
                bias[i] = init_value;
            }
        }
    }
}

size_t Layer::getNumberOfNeurons() const {
    return n_neurons;
}

float Layer::getError() const {
    return error;
}

float* Layer::getNeurons() const {
    return neurons;
}

size_t Layer::getSize() const {
    return n_neurons;
}

void Layer::setDebug(bool value) {
    debug = value;
}

void Layer::setNeurons(const float* data, size_t size){
    std::memcpy(neurons, data, size * sizeof(float));
}

void Layer::forward_pass(const Layer& prev){
    // y = x*w + b
    size_t input_size = prev.getSize();
    float *input = prev.getNeurons();
    for (size_t i=0; i < n_neurons; i++){
        float sum = 0;
        if (debug) std::cout<< "Neuron " << i << ": ";
        for (size_t j=0; j < input_size; j++){
            if (debug){
                std::cout << weights[i*input_size + j] << "x" << input[j] << \
                    " + ";
            }
            sum += (weights[i*input_size + j] * input[j]);
        }
        sum += bias[i];
        if (debug) std::cout << bias[i] << " = " << sum << "\n";
        neurons[i] = ActivationFuncs::ReLU(sum);
    }
}

void Layer::calcError(float* target){
    error = CostFuncs::MSE(neurons, target, n_neurons);
}

void Layer::backward_pass(const Layer& output){

}

Layer::~Layer(){
    if (weights!=nullptr) delete[] weights; 
    if (bias!=nullptr) delete[] bias;
    if (neurons!=nullptr) delete[] neurons;
    number_layers -= 1;
}

