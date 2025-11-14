find_package(GTest REQUIRED)

add_executable(tests layers/DenseLayerTest.cpp)

target_include_directories(tests PRIVATE ../src)
target_link_libraries(tests PRIVATE Eigen3::Eigen GTest::GTest GTest::Main)
target_sources(tests PRIVATE ../src/layers/DenseLayer.cpp)

add_custom_command(TARGET tests
    POST_BUILD
    COMMAND tests
)

enable_testing()
add_test(NAME DenseLayerTest COMMAND tests)