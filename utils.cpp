#include <iostream>
#include "utils.h"

std::vector<double> elementWiseMultiplication(const std::vector<double>& a, const std::vector<double>&b){
    std::vector<double> result;
    for(size_t i = 0; i < a.size(); i++){
        result.push_back(a[i] * b[i]);
    }
    return result;
}

std::vector<double> scalarVectorMultiplication(const std::vector<double>& a, double scalar){
    std::vector<double> result;
    for(size_t i = 0; i <a.size(); i++){
        result.push_back(a[i] * scalar);
    }
    return result;
}

std::vector<double> add(const std::vector<double>& a, const std::vector<double>&b){
    std::vector<double> result;
    for(size_t i = 0; i < a.size(); i++){
        result.push_back(a[i] + b[i]);
    }
    return result;
}

std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>&b){
    std::vector<double> result;
    for(size_t i = 0; i < a.size(); i++){
        result.push_back(a[i] - b[i]);
    }
    return result;
}

double dotProduct(const std::vector<double>& a, const std::vector<double>&b){
    double result = 0;
    for(size_t i = 0; i < a.size(); i++){
        result += a[i] * b[i];
    }
    return result;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>>& matrix){
    if(matrix.empty()) return {};
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols){
    std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            weights[i][j] = dist(gen);
        }
    }
    return weights;
}

std::vector<double> biasInitializer(int size){
    std::vector<double> bias(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(size_t i = 0; i < size; i++){
        bias[i] = dist(gen);
    }
    return bias;
}