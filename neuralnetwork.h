#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <iostream>
#include <memory>
#include "layer.h"
#include "losses.h"
class NeuralNetwork{
public:
    NeuralNetwork(){
        std::cout<<"Hello"<<std::endl;
    };
    std::vector<std::unique_ptr<Layer>> layers;
    void addLayer(Layer* layer);
    std::vector<double> forwardPropagate(const std::vector<double>& inputData);
    std::vector<double> predict(std::vector<double> input);
    void backwardPropagate(const std::vector<double>& error, double learningRate);
    void train(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, double learningRate, double epochs);
};

#endif