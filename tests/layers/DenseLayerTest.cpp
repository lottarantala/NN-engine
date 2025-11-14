#include <gtest/gtest.h>

#include <layers/DenseLayer.hpp>
#include <Eigen/Dense>

class DenseLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<layer::DenseLayer>(inputSize, outputSize, learningRate);
    }

    const int inputSize = 3;
    const int outputSize = 2;
    const double learningRate = 1e-3;
    std::unique_ptr<layer::DenseLayer> layer;
};

TEST_F(DenseLayerTest, forwardPass)
{
    const Eigen::VectorXd inputs = (Eigen::VectorXd(3) << 1, 2, 3).finished();

    // y1 = w11 * 1 + w12 * 2 + w13 * 3 + b1 -> 1*1 + 1*2 + 1*3 + 1
    // y2 = w21 * 1 + w22 * 2 + w23 * 3 + b2 -> 1*1 + 1*2 + 1*3 + 1
    Eigen::VectorXd expectedOutput(2);
    expectedOutput << 7, 7;

    const auto output = layer->forward(inputs);
    EXPECT_EQ(outputSize, output.size());
    EXPECT_EQ(expectedOutput, output);
}

TEST_F(DenseLayerTest, backwardPass)
{
    const Eigen::VectorXd inputs = (Eigen::VectorXd(3) << 1, 2, 3).finished();
    const Eigen::VectorXd delta = (Eigen::VectorXd(2) << 1, 1).finished();
    const auto output = layer->forward(inputs);
    const auto gradient = layer->backward(delta);

    EXPECT_EQ(inputSize, gradient.size());
}