/*  
    Author - Satyam Gupta 
    Date -  12/25/25
    A layer is a part of the neural network.
    It contains n number of neurons, and a bias
    The value of neurons in layer y with previous layer x 
    having weights w and a bias b is given as x*W + b = y
*/

#include <cstddef>
#pragma once

class Layer {
    private:
        static bool debug;
        size_t n_neurons;
        size_t input_n_neurons=0; // For first layer, no input neurons
        float error;
        // float** weights; Unflattened is slower (Linear layout of memory)
        float *weights = nullptr;
        float *bias = nullptr;
        float *neurons = nullptr;
        static size_t number_layers;
    public:
        /**
         * @brief - Constructor for input layer
         */
        Layer(size_t number_n);

        /**
         * @brief Allocate memory for weights, bias, and the neurons
         * 
         * @param prev The previous layer
         * @param number_n The number of neurons in this layer
         * @param init_value Optional initial value for all weights and bias 
         */
        Layer(const Layer& prev, size_t number_n, float init_value=0.0f);

        /**
         * @brief get the number of neurons in this layer
         * @return size_t
         */
        size_t getSize() const;

        /**
         * @brief Get the neurons in this layer
         *
         * @return The neurons in this layer
         */
        float* getNeurons() const;
        
        /**
         * @brief Gets the number of neurons in this layer
         * 
         * @return The number of neurons in this layer
         */
        size_t getNumberOfNeurons() const;

        float getError() const;
        void setDebug(bool value);

        /**
         * @brief initialize the weights with the given value
         * 
         * @param init_value Optional, If none provided uses random numbers
         */
        void initialize_weights(float init_value=0.0f);

        /**
         * @brief - sets the values of the neurons to the passed array
         * 
         * @param data the data to be copied in the neurons of this layer
         */
        void setNeurons(const float* data, size_t size);
        
        /**
         * @brief Executes the forward pass step
         * 
         * @param prev - Layer to use as the input for this layer
         */
        void forward_pass(const Layer& prev);
        
        /**
         * @brief Executes the backward pass step and updates weights
         * 
         * @param next Uses the next layer to calculate the difference
         */
        void backward_pass(const Layer& next);
        
        /**
         * @brief Calculates the error using the target
         * 
         * @param target The desired values of the neurons
         */
        void calcError(float* target);
        
        /**
         * @brief Destructor to ensure no memory leaks
         */
        ~Layer();
    };