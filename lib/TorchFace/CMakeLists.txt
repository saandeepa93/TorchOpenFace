# Include .h files
SET(
  SOURCE
  src/TorchFaceLandmarkImg.cpp
  src/TorchUtils.cpp
  bindings.cpp
  src/TorchFeatureExtraction.cpp
  )
  
SET(
  HEADERS
  include/TorchFaceLandmarkImg.h
  include/TorchUtils.h
  include/TorchFeatureExtraction.h
)
  
add_library(TorchFace SHARED ${SOURCE} ${HEADERS})
target_include_directories(TorchFace PUBLIC include/)

MESSAGE("TorchFace_SOURCE_DIR: ${TorchFace_SOURCE_DIR}")

# Include OpenCVs .h files
target_include_directories(TorchFace PUBLIC ${OpenCV_INCLUDE_DIRS})
MESSAGE("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
MESSAGE("OpenCV_LIBS: ${OpenCV_LIBS}")

# Link Torchlibs
target_link_libraries(TorchFace ${TORCH_LIBRARIES})
target_link_libraries(TorchFace ${OpenCV_LIBS})
target_link_libraries(TorchFace ${OpenBLAS_LIBRARIES})
# Link other external
target_link_libraries(TorchFace LandmarkDetector)
target_link_libraries(TorchFace FaceAnalyser)
target_link_libraries(TorchFace GazeAnalyser)
target_link_libraries(TorchFace Utilities)
