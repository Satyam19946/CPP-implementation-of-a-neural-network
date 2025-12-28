/*
    Author - Satyam Gupta
    Date - 12/25/25
    Implementation of the Layer class defined in Layer.h
*/

#include"Layer.h"
#include<cstring>
#include<iostream>

size_t Layer::number_layers;
bool Layer::debug = false;

Layer::Layer(size_t n_neurons){
    this->n_neurons = n_neurons;
    neurons = new float[n_neurons];
    number_layers += 1;
}

Layer::Layer(const Layer& prev, size_t n_neurons){
    size_t cols = prev.getSize();
    weights = new float[n_neurons * cols]; // input=4x1, output=6x1 weights=6x4
    bias = new float[n_neurons];
    neurons = new float[n_neurons];
    this->n_neurons = n_neurons;
    number_layers += 1;
}

void Layer::setDebug(bool value) {
    debug = value;
}

size_t Layer::getNumberOfNeurons() const {
    return n_neurons;
}

float* Layer::getNeurons() const {
    return neurons;
}

size_t Layer::getSize() const {
    return n_neurons;
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
        for (size_t j=0; j < input_size; j++){
            if (debug){
                std::cout << weights[i*input_size + j] << "\n" << input[j] << \
                "\n";
            }
            sum += weights[i*input_size + j] * input[j];
        }
        neurons[i] = sum;
    }
}

Layer::~Layer(){
    if (weights!=nullptr) delete[] weights; 
    if (bias!=nullptr) delete[] bias;
    if (neurons!=nullptr) delete[] neurons;
    number_layers -= 1;
}

