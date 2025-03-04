#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.cpp"
#include <vector>
#include <iostream>
class NeuralNetwork{
public:
    std::vector<std::unique_ptr<Layer>> layers;
    void addLayer(Layer* layer);
    std::vector<double> forwardPropagate(const std::vector<double>& inputData);
    void backwardPropagate(const std::vector<double>& error, double learningRate);
    void train(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, double learningRate, double epochs);
};

#endif