#pragma once

#include "Activation.hpp"

#include <Eigen/Dense>

namespace layer {

class SigmoidActivation : public Activation
{
public:
    explicit SigmoidActivation(int inputSize)
        : lastOutput{Eigen::VectorXd::Zero(inputSize)}
    {
    }

    virtual ~SigmoidActivation() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        lastOutput = (1.0 / (1.0 + (-input.array()).exp())).matrix();
        return lastOutput;
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& delta)
    {
        return (delta.array() * lastOutput.array() * (1.0 - lastOutput.array())).matrix();
    }

private:
    Eigen::VectorXd lastOutput;
};

} // namespace layer