/*  
    Author - Satyam Gupta 
    Date -  12/25/25
    A layer is a part of the neural network.
    It contains n number of neurons, and a bias
    The value of neurons in layer y with previous layer x 
    having weights w and a bias b is given as x*W + b = y
*/
#pragma once

#include <cstddef>

class Layer {
    private:
        static bool debug;
        size_t n_neurons;
        size_t input_n_neurons=0; // For first layer, no input neurons
        float error;
        // float** weights; Unflattened is slower (Linear layout of memory)
        float *weights = nullptr;
        // in backprop first all gradients are calc then weights are updated
        float *new_weights = nullptr;
        float *bias = nullptr;
        float *neurons = nullptr;
        float *delta = nullptr;
        static size_t number_layers;
    public:
        /**
         * @brief - Constructor for input layer
         */
        Layer(size_t number_n);

        /**
         * @brief Allocate memory for weights, bias, and the neurons
         * 
         * @param prev The previous layer (TODO: DONT NEED WHOLE LAYER, convert\
         *                                          to Layer* prev)
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
         * @brief Get the weights connecting this layer to the prev layer
         */
        float* getWeights() const;

        /**
         * @brief Get the error in this layer (used for backprop)
         * 
         * @return The derivative of this layer with respect to the target
         */
        float* getDelta() const;

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
         * @brief backward pass for the output layer
         */
        void backward_pass(const float* output, const Layer& input,\
                                            const float learning_rate);

        /**
         * @brief Executes the backward pass step and updates weights
         * 
         * @param output The layer right after this layer
         * @param input The layer before this layer
         * @param learning_rate The learning rate for this function
         */
        void backward_pass(const Layer& output, const Layer& input,\
                                                    const float learning_rate);
        
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