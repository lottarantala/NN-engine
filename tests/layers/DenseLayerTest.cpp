#include <gtest/gtest.h>

#include <layers/DenseLayer.hpp>
#include <Eigen/Dense>

class DenseLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        layer = std::make_unique<layer::DenseLayer>(3, 2);
    }

    std::unique_ptr<layer::DenseLayer> layer;
};

TEST_F(DenseLayerTest, forwardPass)
{
    Eigen::VectorXd inputs(3);
    inputs << 1, 2, 3;

    // y1 = w11 * 1 + w12 * 2 + w13 * 3 + b1 -> 1*1 + 1*2 + 1*3 + 1
    // y2 = w21 * 1 + w22 * 2 + w23 * 3 + b2 -> 1*1 + 1*2 + 1*3 + 1
    Eigen::VectorXd expectedOutput(2);
    expectedOutput << 7, 7;

    const auto output = layer->forward(inputs);
    EXPECT_EQ(2, output.size());
    EXPECT_EQ(expectedOutput, output);
}