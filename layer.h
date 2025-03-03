#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>

class Layer{
public:
    std::vector<double> input;
    std::vector<double> output;
    virtual std::vector<double> forward(const std::vector<double>& inputData) = 0;
    virtual std::vector<double> backward(const std::vector<double>& error, double learningRate) = 0;
    virtual ~Layer(){};
};

#endif