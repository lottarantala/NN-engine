add_library(nn_layers
    DenseLayer.cpp
    activation/ActivationLayer.cpp
)

target_link_libraries(nn_layers PUBLIC Eigen3::Eigen)
target_include_directories(nn_layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/activations)