#pragma once
#include<iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include<vector>


#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>
#include <GazeEstimation.h>

#include <ImageCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>

namespace TorchFaceAnalysis{

  class TorchFace:public torch::CustomClassHolder {
    public:
      TorchFace(std::vector<std::string> arguments);
      std::string getter();
      void setter(std::string root);
      std::string model_root="root_path";

      // OpenFace class Members
      LandmarkDetector::CLNF face_model;
      FaceAnalysis::FaceAnalyser face_analyser;
      LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn;
    private:
  };
}
