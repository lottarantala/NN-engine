#pragma once

#include "Activation.hpp"

#include <Eigen/Dense>

namespace layer {

class TanhActivation : public Activation
{
public:
    explicit TanhActivation(int inputSize)
        : lastInput{Eigen::VectorXd::Zero(inputSize)}
    {
    }

    virtual ~TanhActivation() = default;

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