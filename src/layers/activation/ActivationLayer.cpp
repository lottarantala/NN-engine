#include "ActivationLayer.hpp"
#include "ActivationIfc.hpp"

#include <Eigen/Dense>
#include <memory>

namespace layer {

ActivationLayer::ActivationLayer(std::unique_ptr<ActivationIfc> activationFunction)
    : activationFunction{std::move(activationFunction)}
{
}

Eigen::VectorXd ActivationLayer::forward(const Eigen::VectorXd& input)
{
    return activationFunction->forward(input);
}

Eigen::VectorXd ActivationLayer::backward(const Eigen::VectorXd& delta)
{
    return activationFunction->backward(delta);
}

void ActivationLayer::update()
{
    // no update for activation function
}

} // namespace layer
