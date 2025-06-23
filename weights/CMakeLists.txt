cmake_minimum_required(VERSION 3.12)
project(vad_model_dpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

# Vitis AI includes and libraries
set(VITIS_AI_ROOT $ENV{CONDA_PREFIX})
include_directories(${VITIS_AI_ROOT}/include)
link_directories(${VITIS_AI_ROOT}/lib)

# Add executable
add_executable(vad_model_dpu vad_model_dpu.cpp)

# Link libraries
target_link_libraries(vad_model_dpu
    xir
    vart-runner
    vart-util
    vart-buffer-object
    ${GLOG_LIBRARIES}
)

target_include_directories(vad_model_dpu PRIVATE ${GLOG_INCLUDE_DIRS})
target_compile_options(vad_model_dpu PRIVATE ${GLOG_CFLAGS_OTHER})
