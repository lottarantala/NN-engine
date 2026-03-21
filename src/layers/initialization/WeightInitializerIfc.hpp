#pragma once

#include <Eigen/Dense>

namespace weights {

class WeightInitializerIfc
{
public:
    WeightInitializerIfc() noexcept = default;
    virtual ~WeightInitializerIfc() = default;

    virtual Eigen::MatrixXd initialize(const int rows, const int cols) const = 0;
};

} // namespace weights
