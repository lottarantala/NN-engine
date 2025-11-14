#pragma once

#include "LayerBase.hpp"

#include <Eigen/Dense>

namespace layer {

class DenseLayer : public LayerBase
{
public:
    DenseLayer(int inputSize, int outputSize);
    virtual ~DenseLayer() = default;

    Eigen::VectorXd forward() override;
    Eigen::VectorXd backward() override;
    void update() override;

private:
    int inputSize;
    int outputSize;
};

} // namespace layer