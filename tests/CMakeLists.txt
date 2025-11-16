find_package(GTest REQUIRED)

add_executable(tests layers/DenseLayerTest.cpp layers/ActivationLayerTest.cpp)

target_include_directories(tests PRIVATE ../src)
target_link_libraries(tests PRIVATE nn_layers Eigen3::Eigen GTest::GTest GTest::Main)

add_custom_command(TARGET tests
    POST_BUILD
    COMMAND tests
)

enable_testing()
add_test(NAME DenseLayerTest COMMAND tests)
add_test(NAME ActivationLayerTest COMMAND tests)