#include "DenseLayer.hpp"

#include <Eigen/Dense>

namespace layer {

DenseLayer::DenseLayer(int inputSize, int outputSize, double learningRate)
    : learningRate{learningRate}
    , weights{Eigen::MatrixXd::Random(outputSize, inputSize) * 0.1} // TODO: weight initialization functions
    , bias{Eigen::VectorXd::Ones(outputSize)}
    , intermediateOutput{Eigen::VectorXd::Zero(outputSize)}
    , lastInput{Eigen::VectorXd::Zero(inputSize)}
    , gradWeights{Eigen::MatrixXd::Zero(outputSize, inputSize)}
    , gradBias{Eigen::VectorXd::Zero(outputSize)}
{
}

Eigen::VectorXd DenseLayer::forward(const Eigen::VectorXd& inputs)
{
    lastInput = inputs;
    intermediateOutput = (weights * inputs) + bias;
    return intermediateOutput;
}

Eigen::VectorXd DenseLayer::backward(const Eigen::VectorXd& delta)
{
    gradWeights = delta * lastInput.transpose();
    gradBias = delta;
    return weights.transpose() * delta;
}

void DenseLayer::update()
{
    weights -= learningRate * gradWeights;
    bias -= learningRate * gradBias;
}

} // namespace layer