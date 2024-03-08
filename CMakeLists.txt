cmake_minimum_required(VERSION 3.13)
project(ex3)

# Threw some Dlib error.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
set(CUDNN_LIBRARY /usr/local/cuda/lib64)
set(USE_CUDNN 1)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
# OpenBLAS
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")



# find_package(OpenBLAS REQUIRED)
find_package(OpenBLAS REQUIRED)
if ( ${OpenBLAS_FOUND} )
MESSAGE("OpenBLAS information:")
MESSAGE("  OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")
MESSAGE("OpenBLAS FOUND IN ${OpenBLAS_DIR}")
MESSAGE("OpenBLAS_INCLUDE_DIR: ${OpenBLAS_INCLUDE_DIR}")
else()
MESSAGE(FATAL_ERROR "OpenBLAS not found in the system.")
endif()

# OpenCV
find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect)
MESSAGE("OpenCV FOUND IN ${OpenCV_DIR}")


# BOOST
find_package( Boost 1.5.9 COMPONENTS filesystem system)

# dlib
find_package(dlib 19.13 REQUIRED)
if ( ${dlib_FOUND} )
MESSAGE("DLIB FOUND IN ${dlib_DIR}")
endif()


# LibTorch
set(CMAKE_PREFIX_PATH "/home/r/rra-saandeepaath/.conda/envs/openface/lib/python3.10/site-packages/torch")
# SET(CMAKE_PREFIX_PATH /home/saandeepaath-admin/anaconda3/envs/cpp/lib/python3.10/site-packages/torch)
find_package(Torch REQUIRED)

# LandmarkDetector library
add_subdirectory(lib/LandmarkDetector)
# Facial Expression analysis library
add_subdirectory(lib/FaceAnalyser)
# Gaze estimation library
add_subdirectory(lib/GazeAnalyser)
# Utilities library
add_subdirectory(lib/Utilities)
# OpenFace Torch
add_subdirectory(lib/TorchFace)

# find_package(CUDAToolkit REQUIRED)
# message(STATUS "CUDA Toolkit version detected as ${CUDAToolkit_VERSION}")
# EXE
# add_subdirectory(exe)
