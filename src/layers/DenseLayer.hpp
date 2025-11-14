#pragma once

#include "LayerBase.hpp"

#include <Eigen/Dense>

namespace layer {

class DenseLayer : public LayerBase
{
public:
    DenseLayer(int inputSize, int outputSize, double learningRate);
    virtual ~DenseLayer() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& inputs) override;
    Eigen::VectorXd backward(const Eigen::VectorXd& delta) override;
    void update() override;

private:
    double learningRate;
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;
    Eigen::VectorXd intermediateOutput;
    Eigen::VectorXd lastInput;
    Eigen::MatrixXd gradWeights;
    Eigen::VectorXd gradBias;
};

} // namespace layer