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
    const Eigen::VectorXd expectedOutput = (Eigen::VectorXd(2) << 7, 7).finished();

    const auto output = layer->forward(inputs);
    EXPECT_EQ(outputSize, output.size());
    EXPECT_EQ(expectedOutput, output);
}

TEST_F(DenseLayerTest, backwardPass)
{
    const Eigen::VectorXd inputs = (Eigen::VectorXd(3) << 1, 2, 3).finished();
    const Eigen::VectorXd delta = (Eigen::VectorXd(2) << 2, 3).finished();
    const auto layerOutput = layer->forward(inputs);

    // g1 = w11 * 2 + w21 * 3
    // g2 = w12 * 2 + w22 * 3
    // g3 = w13 * 2 + w23 * 3
    const Eigen::VectorXd expectedOutput = (Eigen::VectorXd(3) << 5, 5, 5).finished();

    const auto gradient = layer->backward(delta);
    EXPECT_EQ(inputSize, gradient.size());
}

TEST_F(DenseLayerTest, twoLayerForwardAndBackwardPass)
{
    auto layer2 = std::make_unique<layer::DenseLayer>(outputSize, outputSize, learningRate);
    const Eigen::VectorXd inputs = (Eigen::VectorXd(3) << 1, 2, 3).finished();
    const auto output1 = layer->forward(inputs); // [7, 7]

    // y1 = y2 = 1*7 + 1*7 + 1
    const Eigen::VectorXd expectedOutput = (Eigen::VectorXd(2) << 15, 15).finished();

    const auto output2 = layer2->forward(output1);
    EXPECT_EQ(outputSize, output2.size());
    EXPECT_EQ(expectedOutput, output2);

    const Eigen::VectorXd delta1 = (Eigen::VectorXd(2) << 1, 5).finished();

    const auto grad1 = layer2->backward(delta1);
    const auto grad2 = layer->backward(grad1);

    // [w11 * 1 + w21 * 5, w21 * 1 + w22 * 5]
    const Eigen::VectorXd expectedGrad1 = (Eigen::VectorXd(2) << 6, 6).finished();

    // [w11 * 6 + w21 * 6, w21 * 6 + w22 * 6, w31 * 6 + w32 * 6]
    const Eigen::VectorXd expectedGrad2 = (Eigen::VectorXd(3) << 12, 12, 12).finished();

    EXPECT_EQ(outputSize, grad1.size());
    EXPECT_EQ(inputSize, grad2.size());
    EXPECT_EQ(expectedGrad1, grad1);
    EXPECT_EQ(expectedGrad2, grad2);
}