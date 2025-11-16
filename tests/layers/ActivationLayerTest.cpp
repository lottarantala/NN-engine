#include <layers/activation/ActivationLayer.hpp>
#include <layers/activation/LeakyReLUActivation.hpp>
#include <layers/activation/ReLUActivation.hpp>
#include <layers/activation/SigmoidActivation.hpp>
#include <layers/activation/TanhActivation.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>

class ActivationLayerTest : public ::testing::Test {
protected:
    const int inputSize = 5;
    const double epsilon = 1e-4;
};

TEST_F(ActivationLayerTest, reluActivation)
{
    auto layer = std::make_unique<layer::ActivationLayer>(std::make_unique<layer::ReLUActivation>(inputSize));

    const Eigen::VectorXd inputs = (Eigen::VectorXd(inputSize) << -1.0, 0.0, 1.0, 2.0, 3.0).finished();
    const Eigen::VectorXd delta = (Eigen::VectorXd(inputSize) << 2.0, 1.0, 3.0, 4.0, -1.0).finished();
    const Eigen::VectorXd expectedOutput = (Eigen::VectorXd(inputSize) << 0.0, 0.0, 1.0, 2.0, 3.0).finished();
    const Eigen::VectorXd expectedGradient = (Eigen::VectorXd(inputSize) << 0.0, 0.0, 3.0, 4.0, -1.0).finished();

    const auto layerOutput = layer->forward(inputs);
    const auto layerGradient = layer->backward(delta);
    EXPECT_EQ(inputSize, layerOutput.size());
    EXPECT_EQ(inputSize, layerGradient.size());
    EXPECT_TRUE(layerOutput.isApprox(expectedOutput, epsilon));
    EXPECT_TRUE(layerGradient.isApprox(expectedGradient, epsilon));
}

TEST_F(ActivationLayerTest, leakyreluActivation)
{
    auto layer = std::make_unique<layer::ActivationLayer>(std::make_unique<layer::LeakyReLUActivation>(inputSize, 0.01));

    const Eigen::VectorXd inputs = (Eigen::VectorXd(inputSize) << -2.0, -1.0, 0.0, 1.0, 2.0).finished();
    const Eigen::VectorXd delta = (Eigen::VectorXd(inputSize) << 2.0, 1.0, 3.0, 4.0, -1.0).finished();
    const Eigen::VectorXd expectedOutput = (Eigen::VectorXd(inputSize) << -0.02, -0.01, 0, 1, 2).finished();
    const Eigen::VectorXd expectedGradient = (Eigen::VectorXd(inputSize) << 0.02, 0.01, 0.03, 4, -1).finished();

    const auto layerOutput = layer->forward(inputs);
    const auto layerGradient = layer->backward(delta);
    EXPECT_EQ(inputSize, layerOutput.size());
    EXPECT_EQ(inputSize, layerGradient.size());
    EXPECT_TRUE(layerOutput.isApprox(expectedOutput, epsilon));
    EXPECT_TRUE(layerGradient.isApprox(expectedGradient, epsilon));
}

TEST_F(ActivationLayerTest, tanhActivation)
{
    auto layer = std::make_unique<layer::ActivationLayer>(std::make_unique<layer::TanhActivation>(inputSize));

    const Eigen::VectorXd inputs = (Eigen::VectorXd(inputSize) << -2.0, -1.0, 0.0, 1.0, 2.0).finished();
    const Eigen::VectorXd delta = (Eigen::VectorXd(inputSize) << 2.0, 1.0, 3.0, 4.0, -1.0).finished();
    const Eigen::VectorXd expectedOutput = (Eigen::VectorXd(inputSize) << -0.9640, -0.7616, 0.0, 0.7616, 0.9640).finished();
    const Eigen::VectorXd expectedGradient = (Eigen::VectorXd(inputSize) << 0.1414, 0.4200, 3.0000, 1.6800, -0.0707).finished();

    const auto layerOutput = layer->forward(inputs);
    const auto layerGradient = layer->backward(delta);
    EXPECT_EQ(inputSize, layerOutput.size());
    EXPECT_EQ(inputSize, layerGradient.size());
    EXPECT_TRUE(layerOutput.isApprox(expectedOutput, epsilon));
    EXPECT_TRUE(layerGradient.isApprox(expectedGradient, epsilon));
}

TEST_F(ActivationLayerTest, sigmoidActivation)
{
    auto layer = std::make_unique<layer::ActivationLayer>(std::make_unique<layer::SigmoidActivation>(inputSize));

    const Eigen::VectorXd inputs = (Eigen::VectorXd(inputSize) << -2.0, -1.0, 0.0, 1.0, 2.0).finished();
    const Eigen::VectorXd delta = (Eigen::VectorXd(inputSize) << 2.0, 1.0, 3.0, 4.0, -1.0).finished();
    const Eigen::VectorXd expectedOutput = (Eigen::VectorXd(inputSize) << 0.1192, 0.2689, 0.5000, 0.7311, 0.8808).finished();
    const Eigen::VectorXd expectedGradient = (Eigen::VectorXd(inputSize) << 0.2100, 0.1966, 0.7500, 0.7864, -0.1050).finished();

    const auto layerOutput = layer->forward(inputs);
    const auto layerGradient = layer->backward(delta);
    EXPECT_EQ(inputSize, layerOutput.size());
    EXPECT_EQ(inputSize, layerGradient.size());
    EXPECT_TRUE(layerOutput.isApprox(expectedOutput, epsilon));
    EXPECT_TRUE(layerGradient.isApprox(expectedGradient, epsilon));
}