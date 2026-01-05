/*  
    Author - Satyam Gupta 
    Date - 1/4/26
    A neural network consists of the following - 
        1. An input layer - reads the data (x)
        2. n number of hidden layers (performs computations on x)
        3. An output layer - used for predictions (y) 
*/

#include "NeuralNetwork.h"
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<iostream>

NeuralNetwork::NeuralNetwork(float learning_rate){
    learning_rate = learning_rate;
}

int NeuralNetwork::add_layer(size_t size){
    if (NeuralNetwork::number_layers == 0){ //this is the input layer
        Layer* x = new Layer(size);
        layers.push_back(x);
    } else {
        Layer* h1 = new Layer(*(layers.at(NeuralNetwork::number_layers-1)), \
                                                        size);
        layers.push_back(h1);
    }
    NeuralNetwork::number_layers += 1;
    return 1;
}

void NeuralNetwork::display_layers(){
    for (size_t i = 0; i < NeuralNetwork::number_layers; i++){
        Layer* curr_layer = layers.at(i);
        std::cout<<"Layer "<<i<<": "<<"has "<<(*curr_layer).getSize()<< \
                                            " neurons\n";
    }
}

int NeuralNetwork::read_input(std::string file_name){
    std::ifstream input_file(file_name); 
    std::string line;
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return 1;
    }

    std::getline(input_file, line); //ignore the header line
    while (std::getline(input_file, line)) {
        std::stringstream ss(line);
        std::string field;
        std::vector<int> current_row_inputs;
        bool is_first_column = true;

        // Split line by comma
        while (std::getline(ss, field, ',')) {
            try {
                int value = std::stoi(field); // Convert string to integer

                if (is_first_column) {
                    targets.push_back(value); // First column is target
                    is_first_column = false;
                } else {
                    current_row_inputs.push_back(value); // Others are inputs
                }
            } catch (const std::exception& e) {
                // Skip headers or invalid numeric strings
                continue; 
            }
        }
        
        if (!current_row_inputs.empty()) {
            inputs.push_back(current_row_inputs);
        }
    }
    return 1;
}

void NeuralNetwork::display_input(size_t size){
    for (size_t i = 0; i < size; i++){
        std::cout<<"Target at "<< i << ": "<< targets.at(i)<<std::endl; 
        for (size_t j = 0; j < 50; j++){
            std::cout<<"Pixel at "<<j<<": "<< inputs.at(i).at(j)<<std::endl;
        }
    }
}

void train(size_t epoch){
    return;
}

size_t NeuralNetwork::number_layers = 0;