#pragma once

#include<cstring>
#include<fstream>
#include"Layer.h"
#include<vector>



class NeuralNetwork{
    float learning_rate;
    static size_t number_layers;
    std::ifstream train_fp, test_fp;
    std::vector<Layer*> layers;
    std::vector<int> targets;
    std::vector<std::vector<int>> inputs;

    public:
        NeuralNetwork(float learning_rate=0.0);
        int add_layer(size_t size);
        int read_input(std::string filename);
        void display_input(size_t size);
        void train(size_t epochs);
        void display_layers();
};