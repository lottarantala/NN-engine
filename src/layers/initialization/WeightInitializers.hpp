#pragma once

#include <Eigen/Dense>
#include <random>

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
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        return Eigen::MatrixXd::Zero(rows, cols);
    }
};

class OneInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        return Eigen::MatrixXd::Ones(rows, cols);
    }
};

class GlorotUniformInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        const double limit = std::sqrt(6.0 / (rows + cols));
        return Eigen::MatrixXd::Random(rows, cols) * limit;
    }
};

class GlorotNormalInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        const double stddev = std::sqrt(2.0 / (rows + cols));
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, stddev);

        Eigen::MatrixXd result(rows, cols);
        result = result.unaryExpr([&](double) { return dist(gen); });
        return result;
    }
};

class RandomUniformInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols, const double min, const double max) const
    {
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(min, max);

        Eigen::MatrixXd result(rows, cols);
        result = result.unaryExpr([&](double) { return dist(gen); });
        return result;
    }

    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        return initialize(rows, cols, 0.0, 1.0);
    }
};

class RandomNormalInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols, const double mean, const double variance) const
    {
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        const double stddev = std::sqrt(variance);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<> dist(mean, stddev);

        Eigen::MatrixXd result(rows, cols);
        result = result.unaryExpr([&](double) { return dist(gen); });
        return result;
    }
    
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        return initialize(rows, cols, 0.0, 1.0);
    }
};

class HeNormalInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        const double stddev = std::sqrt(2.0 / rows);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, stddev);

        Eigen::MatrixXd result(rows, cols);
        result = result.unaryExpr([&](double) { return dist(gen); });
        return result;
    }
};

class HeUniformInitializer : public WeightInitializerIfc
{
public:
    Eigen::MatrixXd initialize(const int rows, const int cols) const override
    {
        if(rows <= 0 || cols <= 0)
            throw std::invalid_argument("Rows and columns must be positive integers.");

        const double limit = std::sqrt(6.0 / rows);
        return Eigen::MatrixXd::Random(rows, cols) * limit;
    }
};

} // namespace weights
