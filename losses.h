#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <cmath>
#include <math.h>
#include "utils.cpp"

double BCELoss(std::vector<double> pred, std::vector<double> target);
std::vector<double> BCELossDerivative(std::vector<double> pred, std::vector<double> target);

double MSELoss(std::vector<double> pred, std::vector<double> target);
std::vector<double> MSELossDerivative(std::vector<double> pred, std::vector<double> target);

#endif