#pragma once

#include "Activation.hpp"

#include <Eigen/Dense>

namespace layer {

class TanhActivation : public Activation
{
public:
    explicit TanhActivation(int inputSize)
        : lastOutput{Eigen::VectorXd::Zero(inputSize)}
    {
    }

    virtual ~TanhActivation() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        lastOutput = input.array().tanh().matrix();
        return lastOutput;
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& delta)
    {
        return (delta.array() * (1.0 - lastOutput.array().square())).matrix();
    }

private:
    Eigen::VectorXd lastOutput;
};

} // namespace layer