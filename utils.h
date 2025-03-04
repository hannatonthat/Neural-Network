#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>

std::vector<double> elementWiseMultiplication(const std::vector<double>& a, const std::vector<double>&b);
std::vector<double> scalarVectorMultiplication(const std::vector<double>& a, double scalar);
std::vector<double> add(const std::vector<double>& a, const std::vector<double>&b);
std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>&b);
double dotProduct(const std::vector<double>& a, const std::vector<double>&b);
std::vector<std::vector<double>> transpose(std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols);
std::vector<double> biasInitializer(int size);

#endif