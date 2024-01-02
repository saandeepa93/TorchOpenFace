SET(
  SOURCE
  src/TorchFace.cpp
)

SET(
  HEADERS
  include/TorchFace.h
)

add_library(TorchFace SHARED ${SOURCE} ${HEADERS})

MESSAGE("TorchFace_SOURCE_DIR: ${TorchFace_SOURCE_DIR}")

# Include .h files
target_include_directories(TorchFace PUBLIC include/)
# Include OpenCVs .h files
target_include_directories(TorchFace PUBLIC ${OpenCV_INCLUDE_DIRS})
MESSAGE("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

# Link Torchlibs
target_link_libraries(TorchFace ${TORCH_LIBRARIES})
# Link other external
target_link_libraries(TorchFace LandmarkDetector)
target_link_libraries(TorchFace FaceAnalyser)
target_link_libraries(TorchFace GazeAnalyser)
target_link_libraries(TorchFace Utilities)
