#include "losses.h"

double BCELoss(std::vector<double> pred, std::vector<double> target){
    double sum = 0;
    for(int i = 0; i < pred.size(); i++){
        sum += target[i] * log(pred[i]) + (1 - target[i]) - log((1 - pred[i]));
    }
    int size = target.size();
    double loss = -(1.0 / size) * sum;
    return loss;
}

std::vector<double> BCELossDerivative(std::vector<double> pred, std::vector<double> target){
    std::vector<double> derivative = {(pred[0] - target[0]) / ((pred[0]) * (1 - pred[0]))};
    return derivative;
}

double MSELoss(std::vector<double> pred, std::vector<double> target){
    double sum = 0;
    for(int i = 0; i < pred.size(); i++){
        sum += pow(target[i] - pred[i], 2.0);
    }
    int size = target.size();
    double loss = (1.0 / size) * sum;
    return loss;
}

std::vector<double> MSELossDerivative(std::vector<double> pred, std::vector<double> target){
    std::vector<double> sub = subtract(pred, target);
    std::vector<double> derivative = scalarVectorMultiplication(sub, 2);
    return derivative;
}