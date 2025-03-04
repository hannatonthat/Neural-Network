#include "neuralnetwork.h"
#include "losses.cpp"

void NeuralNetwork::addLayer(Layer* layer){
    layers.emplace_back(layer);
}

std::vector<double> NeuralNetwork::forwardPropagate(const std::vector<double>& inputData){
    std::vector<double> output = inputData;
    for(const auto& layer:layers){
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::backwardPropagate(const std::vector<double>& error, double learningRate){
    std::vector<double> gradError = error;
    for(auto it = layers.rbegin(); it != layers.rend(); it++){
        gradError = (*it)->backward(error, learningRate);
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y, double learningRate, double epochs){
    for(int epoch = 0; epoch < epochs; epoch++){
        double totalLoss = 0.0;
        for(size_t i = 0; i < x.size(); i++){
            std::vector<double> output = forwardPropagate(x[i]);
            double loss = BCELoss(y[i], output);
            totalLoss += loss;
            std::vector<double> lossDerivative = BCELossDerivative(y[i], output);
            backwardPropagate(lossDerivative, learningRate);
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << totalLoss / x.size() << std::endl;
    }
}