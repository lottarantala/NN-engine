add_library(nn_layers INTERFACE)

target_link_libraries(nn_layers INTERFACE Eigen3::Eigen)
target_include_directories(nn_layers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})