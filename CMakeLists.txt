cmake_minimum_required(VERSION 3.18)
project(gravitational-box LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(gravbox
    main.cu
    gl_interop.cpp
    cuda_kernel.cu
)

set_target_properties(gravbox PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gravbox PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(gravbox
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    curand
)
