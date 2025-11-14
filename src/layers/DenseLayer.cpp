#include "DenseLayer.hpp"

#include <Eigen/Dense>

namespace layer {

DenseLayer::DenseLayer(int inputSize, int outputSize)
    : inputSize{inputSize}
    , outputSize{outputSize}
{}

Eigen::VectorXd DenseLayer::forward()
{
    return {};
}

Eigen::VectorXd DenseLayer::backward()
{
    return {};
}

void DenseLayer::update()
{
}

} // namespace layer