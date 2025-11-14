#pragma once

#include <Eigen/Dense>

namespace layer {

class LayerBase
{
public:
    LayerBase() = default;
    virtual ~LayerBase() = default;

    LayerBase(const LayerBase&) = delete;
    LayerBase& operator=(const LayerBase&) = delete;

    LayerBase(LayerBase&&) = default;
    LayerBase& operator=(LayerBase&&) = default;

    virtual Eigen::VectorXd forward() = 0;
    virtual Eigen::VectorXd backward() = 0;
    virtual void update() = 0;
};

} // namespace layer