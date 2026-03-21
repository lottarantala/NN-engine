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

class ZeroInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        return Eigen::MatrixXd::Zero(rows, cols);
    }
};

class OneInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        return Eigen::MatrixXd::Ones(rows, cols);
    }
};

} // namespace weights
