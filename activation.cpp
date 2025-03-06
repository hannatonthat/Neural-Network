#include "activation.h"

std::vector<double> vectSigmoid(const std::vector<double>& x){
    std::vector<double> result;
    for(double i : x){
        result.push_back(1.0 / (1.0 + exp(-i)));
    }
    return result;
}

std::vector<double> vectSigmoidDerivative(const std::vector<double>& x){
    std::vector<double> result;
    std::vector<double> sigmoidX = vectSigmoid(x);
    for(double i : sigmoidX){
        result.push_back(i * (1 - i));
    }
    return result;
}

std::vector<double> vectTanh(const std::vector<double>& x){
     std::vector<double> result;
     for(double i : x){
        result.push_back(tanh(i));
     }
     return result;
}

std::vector<double> vectTanhDerivative(const std::vector<double>& x){
    std::vector<double> result;
    for(double i : x){
        double tanhI = tanh(i);
        result.push_back(1 - tanhI * tanhI);
    }
    return result;
}

std::vector<double> vectReLu(const std::vector<double>& x){
    std::vector<double> result;
    for(double i : x){
        result.push_back(i > 0 ? i : 0);
    }
    return result;
}

std::vector<double> vectReLuDerivative(const std::vector<double>& x){
    std::vector<double> result;
    for(double i : x){
        result.push_back(i > 0 ? 1 : 0);
    }
    return result;
}

std::vector<double> vectLeakyReLu(const std::vector<double>& x){
    std::vector<double> result;
    for(double i : x){
        result.push_back(i > 0 ? i : 0.01 * i);
    }
    return result;
}

std::vector<double> vectLeakyReLuDerivative(const std::vector<double>& x){
    std::vector<double> result;
    for(double i : x){
        result.push_back(i > 0 ? 1 : 0.01);
    }
    return result;
}