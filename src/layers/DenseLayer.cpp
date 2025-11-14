#include "DenseLayer.hpp"

#include <Eigen/Dense>

namespace layer {

DenseLayer::DenseLayer(int inputSize, int outputSize)
    : inputSize{inputSize}
    , outputSize{outputSize}
    , weights{Eigen::MatrixXd::Ones(outputSize, inputSize)} // TODO: weight initialization functions
    , bias{Eigen::VectorXd::Ones(outputSize)}
    , intermediateOutput{Eigen::VectorXd::Zero(outputSize)}
{
}

Eigen::VectorXd DenseLayer::forward(const Eigen::VectorXd& inputs)
{
    intermediateOutput = (weights * inputs) + bias;
    return intermediateOutput;
}

Eigen::VectorXd DenseLayer::backward()
{
    return {};
}

void DenseLayer::update()
{
}

} // namespace layer