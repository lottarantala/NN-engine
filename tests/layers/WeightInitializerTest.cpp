#include <layers/initialization/WeightInitializers.hpp>

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <memory>

namespace {
class WeightInitializerTest : public ::testing::Test
{
protected:
    const int rows = 374;
    const int cols = 441;
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

TEST_F(WeightInitializerTest, zeroInitializerInvalidDimensions)
{
    weights::ZeroInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
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

TEST_F(WeightInitializerTest, oneInitializerInvalidDimensions)
{
    weights::OneInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
}

TEST_F(WeightInitializerTest, glorotUniformInitializer)
{
    weights::GlorotUniformInitializer initializer;

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    const double limit = std::sqrt(6.0 / (rows + cols));
    EXPECT_TRUE((output.array() >= -limit).all());
    EXPECT_TRUE((output.array() <= limit).all());
}

TEST_F(WeightInitializerTest, glorotUniformInitializerInvalidDimensions)
{
    weights::GlorotUniformInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
}

TEST_F(WeightInitializerTest, glorotNormalInitializer)
{
    weights::GlorotNormalInitializer initializer;

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    const double stddev = std::sqrt(2.0 / (rows + cols));
    const double mean = output.mean();
    const double variance = (output.array() - mean).square().sum() / (rows * cols);

    // higher tolerance due to small sample size and randomness
    const double tolerance = 0.1;

    EXPECT_NEAR(mean, 0.0, tolerance);
    EXPECT_NEAR(variance, stddev * stddev, tolerance);
}

TEST_F(WeightInitializerTest, glorotNormalInitializerInvalidDimensions)
{
    weights::GlorotNormalInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
}

TEST_F(WeightInitializerTest, randomUniformInitializerDefault)
{
    weights::RandomUniformInitializer initializer;
    const double min = 0.0;
    const double max = 1.0;

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    EXPECT_TRUE((output.array() >= min).all());
    EXPECT_TRUE((output.array() <= max).all());
}

TEST_F(WeightInitializerTest, randomUniformInitializerCustom)
{
    weights::RandomUniformInitializer initializer;
    const double min = -1.8;
    const double max = 33.4;

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    EXPECT_TRUE((output.array() >= min).all());
    EXPECT_TRUE((output.array() <= max).all());
}

TEST_F(WeightInitializerTest, randomUniformInitializerInvalidDimensions)
{
    weights::RandomUniformInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
}

TEST_F(WeightInitializerTest, randomNormalInitializerDefault)
{
    weights::RandomNormalInitializer initializer;

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    const double mean = output.mean();
    const double variance = (output.array() - mean).square().sum() / (rows * cols);

    const double expectedMean = 0.0;
    const double expectedVariance = 1.0;
    // higher tolerance due to small sample size and randomness
    const double tolerance = 0.1;

    EXPECT_NEAR(mean, expectedMean, tolerance);
    EXPECT_NEAR(variance, expectedVariance, tolerance);
}

TEST_F(WeightInitializerTest, randomNormalInitializerCustom)
{
    weights::RandomNormalInitializer initializer;
    const double expectedMean = 5.2;
    const double expectedVariance = 2.7;
    // higher tolerance due to small sample size and randomness
    const double tolerance = 0.1;

    const auto output = initializer.initialize(rows, cols, expectedMean, expectedVariance);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    const double mean = output.mean();
    const double variance = (output.array() - mean).square().sum() / (rows * cols);

    EXPECT_NEAR(mean, expectedMean, tolerance);
    EXPECT_NEAR(variance, expectedVariance, tolerance);
}

TEST_F(WeightInitializerTest, randomNormalInitializerInvalidDimensions)
{
    weights::RandomNormalInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
}

TEST_F(WeightInitializerTest, heUniformInitializer)
{
    weights::HeUniformInitializer initializer;

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    const double limit = std::sqrt(6.0 / rows);
    EXPECT_TRUE((output.array() >= -limit).all());
    EXPECT_TRUE((output.array() <= limit).all());
}

TEST_F(WeightInitializerTest, heUniformInvalidDimensions)
{
    weights::HeNormalInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
}

TEST_F(WeightInitializerTest, heNormalInitializer)
{
    weights::HeNormalInitializer initializer;

    const auto output = initializer.initialize(rows, cols);
    ASSERT_EQ(rows, output.rows());
    ASSERT_EQ(cols, output.cols());

    const double stddev = std::sqrt(2.0 / rows);
    const double mean = output.mean();
    const double variance = (output.array() - mean).square().sum() / (rows * cols);

    // higher tolerance due to small sample size and randomness
    const double tolerance = 0.1;

    EXPECT_NEAR(mean, 0.0, tolerance);
    EXPECT_NEAR(variance, stddev * stddev, tolerance);
}

TEST_F(WeightInitializerTest, heNormalInitializerInvalidDimensions)
{
    weights::HeNormalInitializer initializer;
    EXPECT_THROW(initializer.initialize(-1, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, -1), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(0, cols), std::invalid_argument);
    EXPECT_THROW(initializer.initialize(rows, 0), std::invalid_argument);
}

}// namespace
