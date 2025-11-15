#pragma once

#include "Activation.hpp"

#include <Eigen/Dense>

namespace layer {

class LeakyReLUActivation : public Activation
{
public:
    explicit LeakyReLUActivation(int inputSize, double alpha)
        : lastInput{Eigen::VectorXd::Zero(inputSize)}
        , alpha{alpha}
    {
    }

    virtual ~LeakyReLUActivation() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        lastInput = input;
        return (input.array() > 0).select(input, alpha * input).matrix();
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& delta)
    {
        return (lastInput.array() > 0).select(delta, alpha * delta).matrix();
    }

private:
    Eigen::VectorXd lastInput;
    double alpha;
};

} // namespace layer