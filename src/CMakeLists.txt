find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(nn_main
    main.cpp
    layers/DenseLayer.cpp
)

add_subdirectory(layers)

target_include_directories(nn_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nn_main PRIVATE Eigen3::Eigen nn_layers)