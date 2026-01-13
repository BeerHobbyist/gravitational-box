cmake_minimum_required(VERSION 3.18)
project(gravitational-box LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(gravbox_gpu
    gpu/main.cu
    gpu/gl_interop.cpp
    gpu/cuda_kernel.cu
)

set_target_properties(gravbox_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gravbox_gpu PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu
)

target_link_libraries(gravbox_gpu
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
    curand
)

add_executable(gravbox_cpu
    cpu/main.cpp
    cpu/gl_pbo.cpp
    cpu/api.cpp
    cpu/simulation.cpp
    cpu/renderer.cpp
)

target_include_directories(gravbox_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu
)

target_link_libraries(gravbox_cpu
    ${OPENGL_LIBRARIES}
    GLEW::GLEW
    glfw
)
