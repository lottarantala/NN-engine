#pragma once

#include "Activation.hpp"

#include <Eigen/Dense>

namespace layer {

class SigmoidActivation : public Activation
{
public:
    explicit SigmoidActivation(int inputSize)
        : lastInput{Eigen::VectorXd::Zero(inputSize)}
    {
    }

    virtual ~SigmoidActivation() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        lastInput = input;
        return {};
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& delta)
    {
        return {};
    }

private:
    Eigen::VectorXd lastInput;
};

} // namespace layer