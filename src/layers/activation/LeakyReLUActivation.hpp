#pragma once

#include "Activation.hpp"

#include <Eigen/Dense>

namespace layer {

class LeakyReLUActivation : public Activation
{
public:
    explicit LeakyReLUActivation(int inputSize, double delta)
        : lastInput{Eigen::VectorXd::Zero(inputSize)}
        , delta{delta}
    {
    }

    virtual ~LeakyReLUActivation() = default;

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
    double delta;
};

} // namespace layer