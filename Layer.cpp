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
    delta = new float[n_neurons];
    new_weights = new float[n_neurons*input_n_neurons];
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

float* Layer::getWeights() const {
    return weights;
}

size_t Layer::getSize() const {
    return n_neurons;
}

float* Layer::getDelta() const {
    return delta;
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

void Layer::backward_pass(const float* output, const Layer& input, \
                                                const float learning_rate){
    size_t size = this->getSize();
    
    for (size_t i=0; i < size; i++){
        if (debug) std::cout<<"Delta "<<i<< " was: "<<delta[i];
        delta[i] =  output[i] * \
            ActivationFuncs::derivate_ReLU(neurons[i]);
        if (debug) std::cout<<" Delta after update: "<<delta[i]<<"\n";
    }

    float *input_neurons = input.getNeurons();
    size_t input_size = input.getSize();
    for (size_t i=0; i < size; i++){
        for (size_t j=0; j < input_size; j++){
            if (debug){
                std::cout<<"Weight "<<i*size+j<<" was: "<<weights[i*size + j]<<\
                " New Weight = "<<weights[i*size+j]<<" - "<<learning_rate<<\
                "*"<<delta[i]<<"*"<<input_neurons[j] <<" = ";
                
            }
            new_weights[i*input_size + j] -= learning_rate*delta[i]\
                                                *input_neurons[j];
            if (debug) std::cout<<new_weights[i*input_size + j]<<"\n";
        }
        bias[i] -= learning_rate*delta[i];
        if (debug) std::cout<<"New Bias: "<<bias[i]<<"\n";
    }
}

void Layer::backward_pass(const Layer& output, const Layer& input, \
                                                const float learning_rate) {
    size_t size = this->getSize();
    size_t output_size = output.getSize();
    float* output_delta = output.getDelta();
    float* output_weights = output.getWeights();
    float *input_neurons = input.getNeurons();
    size_t input_size = input.getSize();

    for (size_t i=0; i < size; i++){
        float error_from_outputs = 0;
        for (size_t j=0; j < output_size; j++){
            error_from_outputs += output_delta[j] * output_weights[j*size + i];
        }
        if (debug) std::cout<<"Delta "<<i<< " was: "<<delta[i];
        delta[i] =  error_from_outputs * \
            ActivationFuncs::derivate_ReLU(neurons[i]);
        if (debug) std::cout<<" Delta after update: "<<delta[i]<<"\n";
    }

    for (size_t i=0; i < size; i++){
        for (size_t j=0; j < input_size; j++){
            if (debug){
                std::cout<<"Weight "<<i*size+j<<" was: "<<weights[i*size + j]<<\
                " New Weight = "<<weights[i*size+j]<<" - "<<learning_rate<<\
                "*"<<delta[i]<<"*"<<input_neurons[j] <<" = ";
                
            }
            new_weights[i*input_size + j] -= learning_rate*delta[i]*input_neurons[j];
            if (debug) std::cout<<new_weights[i*input_size + j]<<"\n";
        }
        bias[i] -= learning_rate*delta[i];
        if (debug) std::cout<<"New Bias: "<<bias[i]<<"\n";
    }
}

Layer::~Layer(){
    if (weights!=nullptr) delete[] weights; 
    if (bias!=nullptr) delete[] bias;
    if (neurons!=nullptr) delete[] neurons;
    if (delta!=nullptr) delete[] delta;
    if (new_weights!=nullptr) delete[] new_weights;
    number_layers -= 1;
}

