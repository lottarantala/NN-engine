#pragma once

#include <Eigen/Dense>

namespace layer {

class ActivationIfc
{
public:
    ActivationIfc() noexcept = default;
    virtual ~ActivationIfc() = default;
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& inputs) = 0;
    virtual Eigen::VectorXd backward(const Eigen::VectorXd& delta) = 0;
};

} // namespace layer
