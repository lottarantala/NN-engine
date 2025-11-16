#include <layers/DenseLayer.hpp>
#include <layers/activation/ActivationLayer.hpp>
#include <layers/activation/TanhActivation.hpp>

#include <Eigen/Dense>
#include <iostream>
#include <memory>

int main()
{
    Eigen::VectorXd inputs = (Eigen::VectorXd(10) << 0.5, -1.2, 2.0, 0.0, 1.5, -0.7, 0.3, 2.1, -1.0, 0.8).finished();
    Eigen::VectorXd target = (Eigen::VectorXd(2) << 1.0, -0.5).finished();

    auto layer1 = std::make_unique<layer::DenseLayer>(10, 5, 1e-3);
    auto activation1 = std::make_unique<layer::ActivationLayer>(std::make_unique<layer::TanhActivation>(5));
    auto layer2 = std::make_unique<layer::DenseLayer>(5, 2, 1e-3);
    auto activation2 = std::make_unique<layer::ActivationLayer>(std::make_unique<layer::TanhActivation>(2));

    std::cout << "Starting training..." << std::endl;
    for(int i = 0; i < 25; i++)
    {
        std::cout << "Iteration " << i+1 << std::endl;
        // forward pass
        auto output1 = layer1->forward(inputs);
        auto act1 = activation1->forward(output1);
        auto output2 = layer2->forward(act1);
        auto act2 = activation2->forward(output2);
        std::cout << "Output: " << act2.transpose() << std::endl;

        // MSE
        auto loss = (act2 - target).squaredNorm() / act2.size();
        std::cout << "Loss: " << loss << std::endl;

        // backward pass
        auto grad_activation2 = 2.0 * (act2 - target) / act2.size(); // MSE derivative
        auto grad_layer2 = layer2->backward(grad_activation2);
        auto grad_activation1 = activation1->backward(grad_layer2);
        auto grad_layer1 = layer1->backward(grad_activation1);

        layer1->update();
        layer2->update();
    }
    return 0;
}
