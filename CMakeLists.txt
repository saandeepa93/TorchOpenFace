cmake_minimum_required(VERSION 3.13)
project(ex3)

# Threw some Dlib error.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
# OpenBLAS
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
find_package(OpenBLAS REQUIRED)
if ( ${OpenBLAS_FOUND} )
    MESSAGE("OpenBLAS information:")
    MESSAGE("  OpenBLAS_LIBRARIES: ${OpenBLAS_LIB}")
else()
    MESSAGE(FATAL_ERROR "OpenBLAS not found in the system.")
endif()

# OpenCV
# find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect)

set(OpenCV_DIR "/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external/lib/cmake/opencv4/")
find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect
          PATHS "/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external/lib/cmake/opencv4/" NO_DEFAULT_PATH )
MESSAGE("OpenCV FOUND IN ${OpenCV_DIR}")


# BOOST
find_package( Boost 1.5.9 COMPONENTS filesystem system)

# dlib
# set(dlib_DIR /home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external//usr/local/lib/)
find_package(dlib 19.13 REQUIRED 
    PATHS "/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external/usr/local/lib/cmake/dlib" NO_DEFAULT_PATH)
# find_package(dlib 19.13 REQUIRED)
if ( ${dlib_FOUND} )
    MESSAGE("DLIB FOUND IN ${dlib_DIR}")
endif()

# LibTorch
SET(CMAKE_PREFIX_PATH /home/saandeepaath-admin/projects/learning/cpp_cmake/example3/pytorch)
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

# EXE
add_subdirectory(exe)
