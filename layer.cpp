#include "layer.h"

std::vector<double> Sigmoid::forward(const std::vector<double>& inputData){
    input = inputData;
    output = vectSigmoid(input);
    return output;
}

std::vector<double> Sigmoid::backward(const std::vector<double>& error, double learningRate){
    std::vector<double> derivative = vectSigmoidDerivative(input);
    std::vector<double> gradInput;
    for(size_t i = 0; i < derivative.size(); i++){
        gradInput.push_back(derivative[i] * error[i]);
    }
    return gradInput;
}

std::vector<double> Tanh::forward(const std::vector<double>& inputData){
    input = inputData;
    output = vectTanh(input);
    return output;
}

std::vector<double> Tanh::backward(const std::vector<double>& error, double learningRate){
    std::vector<double> derivative = vectTanhDerivative(input);
    std::vector<double> gradInput;
    for(size_t i = 0; i < derivative.size(); i++){
        gradInput.push_back(derivative[i] * error[i]);
    }
    return gradInput;
}

std::vector<double> ReLu::forward(const std::vector<double>& inputData){
    input = inputData;
    output = vectReLu(input);
    return output;
}

std::vector<double> ReLu::backward(const std::vector<double>& error, double learningRate){
    std::vector<double> derivative = vectReLuDerivative(input);
    std::vector<double> gradInput;
    for(size_t i = 0; i < derivative.size(); i++){
        gradInput.push_back(derivative[i] * error[i]);
    }
    return gradInput;
}

std::vector<double> LeakyReLu::forward(const std::vector<double>& inputData){
    input = inputData;
    output = vectLeakyReLu(input);
    return output;
}

std::vector<double> LeakyReLu::backward(const std::vector<double>& error, double learningRate){
    std::vector<double> derivative = vectLeakyReLuDerivative(input);
    std::vector<double> gradInput;
    for(size_t i = 0; i < derivative.size(); i++){
        gradInput.push_back(derivative[i] * error[i]);
    }
    return gradInput;
}

Linear::Linear(int inputSize, int outputSize){
    weights = uniformWeightInitializer(outputSize, inputSize);
    bias = biasInitializer(outputSize);
}

std::vector<double> Linear::forward(const std::vector<double>& inputData){
    input = inputData;
    output.clear();
    output.resize(bias.size());
    for(int i = 0; i < output.size(); i++){
        output[i] = dotProduct(weights[i], input) + bias[i];
    }
    return output;
}

std::vector<double> Linear::backward(const std::vector<double>& error, double learningRate){
    std::vector<double> inputError;
    std::vector<std::vector<double>> weightError;
    std::vector<double> biasError;
    std::vector<std::vector<double>> weightTranspose;
    inputError.clear();
    weightError.clear();
    biasError.clear();
    weightTranspose = transpose(weights);
    biasError = error;
    for (int i = 0; i < weightTranspose.size(); i++){
        inputError.push_back(dotProduct(weightTranspose[i], error));
    }
    for (int j = 0; j < error.size(); j++){
        std::vector<double> row;
        for (int i = 0; i < input.size(); i++) {
            row.push_back(error[j] * input[i]);
        }
        weightError.push_back(row);
    }
    std::vector<double> deltaBias = scalarVectorMultiplication(biasError, learningRate);
    bias = subtract(bias, deltaBias);
    for (int i = 0; i < weightError.size(); i++) {
        std::vector<double> deltaWeight = scalarVectorMultiplication(weightError[i], learningRate);
        weights[i] = subtract(weights[i], deltaWeight);
    }
    return inputError;
}