#pragma once

#include "Activation.hpp"

#include <Eigen/Dense>

namespace layer {

class ReLUActivation : public Activation
{
public:
    explicit ReLUActivation(int inputSize)
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
        return (lastInput.array() > 0).select(delta, 0);
    }

private:
    Eigen::VectorXd lastInput;
};

} // namespace layer