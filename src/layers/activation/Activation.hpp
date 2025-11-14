#pragma once

#include <Eigen/Dense>

namespace layer {

class Activation
{
public:
    virtual ~Activation() = default;
    virtual Eigen::VectorXd forward(const Eigen::VectorXd& inputs) = 0;
    virtual Eigen::VectorXd backward(const Eigen::VectorXd& delta) = 0;
};

} // namespace layer