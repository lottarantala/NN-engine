#pragma once

#include "LayerBase.hpp"

#include <Eigen/Dense>

namespace layer {

class DenseLayer : public LayerBase
{
public:
    DenseLayer(int inputSize, int outputSize);
    virtual ~DenseLayer() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& inputs) override;
    Eigen::VectorXd backward() override;
    void update() override;

private:
    int inputSize;
    int outputSize;
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;
    Eigen::VectorXd intermediateOutput;
};

} // namespace layer