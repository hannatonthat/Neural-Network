#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cmath>

std::vector<double> vectSigmoid(const std::vector<double>& x);
std::vector<double> vectSigmoidDerivative(const std::vector<double>& x);

std::vector<double> vectTanh(const std::vector<double>& x);
std::vector<double> vectTanhDerivative(const std::vector<double>& x);

std::vector<double> vectReLu(const std::vector<double>& x);
std::vector<double> vectReLuDerivative(const std::vector<double>& x);

std::vector<double> vectLeakyReLu(const std::vector<double>& x, double alpha = 0.01);
std::vector<double> vectLeakyReLuDerivative(const std::vector<double>& x, double alpha = 0.01);

#endif