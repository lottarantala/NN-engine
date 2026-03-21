#include <layers/initialization/WeightInitializers.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>

namespace {
class WeightInitializerTest : public ::testing::Test
{
protected:
    const int rows = 3;
    const int cols = 4;
    const double epsilon = 1e-4;
};

TEST_F(WeightInitializerTest, zeroInitializer)
{
    weights::ZeroInitializer initializer;
    const Eigen::MatrixXd expectedOutput = Eigen::MatrixXd::Zero(rows, cols);

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());
    EXPECT_TRUE(output.isApprox(expectedOutput, epsilon));
}

TEST_F(WeightInitializerTest, oneInitializer)
{
    weights::OneInitializer initializer;
    const Eigen::MatrixXd expectedOutput = Eigen::MatrixXd::Ones(rows, cols);

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());
    EXPECT_TRUE(output.isApprox(expectedOutput, epsilon));
}
}// namespace
