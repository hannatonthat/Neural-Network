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

class Sigmoid : public Layer{
public:
    std::vector<double> forward(const std::vector<double>& inputData) override;
    std::vector<double> backward(const std::vector<double>& error, double learningRate) override;
};

class Tanh : public Layer{
public:
    std::vector<double> forward(const std::vector<double>& inputData) override;
    std::vector<double> backward(const std::vector<double>& error, double learningRate) override;
};

class ReLu : public Layer{
public:
    std::vector<double> forward(const std::vector<double>& inputData) override;
    std::vector<double> backward(const std::vector<double>& error, double learningRate) override;
};

class LeakyReLu : public Layer{
public:
    double alpha = 0.01;
    std::vector<double> forward(const std::vector<double>& inputData) override;
    std::vector<double> backward(const std::vector<double>& error, double learningRate) override;
};

class Linear : public Layer{
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    Linear(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double>& inputData) override;
    std::vector<double> backward(const std::vector<double>& error, double learningRate) override;
};

#endif