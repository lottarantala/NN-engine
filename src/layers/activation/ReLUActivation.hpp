#pragma once

#include "ActivationIfc.hpp"

#include <Eigen/Dense>

namespace layer {

class ReLUActivation : public ActivationIfc
{
public:
    explicit ReLUActivation(const int inputSize)
        : lastInput{Eigen::VectorXd::Zero(inputSize)}
    {
    }

    virtual ~ReLUActivation() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        lastInput = input;
        return input.cwiseMax(0.0);
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& delta)
    {
        return (lastInput.array() > 0).select(delta, 0).matrix();
    }

private:
    Eigen::VectorXd lastInput;
};

} // namespace layer
