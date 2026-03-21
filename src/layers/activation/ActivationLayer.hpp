#pragma once

#include "LayerBase.hpp"

#include <Eigen/Dense>
#include <memory>

namespace layer {
class ActivationIfc;

class ActivationLayer : public LayerBase
{
public:
    ActivationLayer(std::unique_ptr<ActivationIfc> activationFunction);
    virtual ~ActivationLayer() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& inputs) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& delta) override;
    void update() override;

private:
    std::unique_ptr<ActivationIfc> activationFunction;
};

} // namespace layer
