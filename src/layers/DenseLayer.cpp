#include "DenseLayer.hpp"

#include <initialization/WeightInitializers.hpp>
#include <Eigen/Dense>

namespace layer {

DenseLayer::DenseLayer(const int inputSize, const int outputSize, const double learningRate, const weights::WeightInitializerIfc& weightInitializer)
    : learningRate{learningRate}
    , weights{weightInitializer.initialize(outputSize, inputSize)}
    , bias{Eigen::VectorXd::Ones(outputSize)}
    , intermediateOutput{Eigen::VectorXd::Zero(outputSize)}
    , lastInput{Eigen::VectorXd::Zero(inputSize)}
    , gradWeights{Eigen::MatrixXd::Zero(outputSize, inputSize)}
    , gradBias{Eigen::VectorXd::Zero(outputSize)}
{
}

DenseLayer::DenseLayer(const int inputSize, const int outputSize, const double learningRate)
    : DenseLayer(inputSize, outputSize, learningRate, weights::ZeroInitializer{})
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
