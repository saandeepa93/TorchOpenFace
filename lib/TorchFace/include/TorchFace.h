#pragma once
#include<iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <map>
#include <any>
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include<vector>
#include "ImageManipulationHelpers.h"

// #include <ATen/core/Dict.h>
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
      torch::Tensor  ExtractFeatures(torch::Tensor img, c10::Dict<std::string, c10::IValue> misc_args);
      void SetImageParams(cv::Mat img);
      std::vector<cv::Rect_<float> > FaceDetection(const cv::Mat_<uchar>& grayscale_image, const cv::Mat& rgb_image, \
            const  c10::Dict<std::string, c10::IValue>& misc_args);

      bool has_bounding_boxes = false;
      std::vector<std::string> arguments;
      
      // OpenFace class Members
      Utilities::RecorderOpenFaceParameters recording_params;
      Utilities::RecorderOpenFace open_face_rec;
      LandmarkDetector::FaceModelParameters det_parameters;
      Utilities::ImageCapture image_reader;
      LandmarkDetector::CLNF face_model;
      FaceAnalysis::FaceAnalyser face_analyser;
      cv::CascadeClassifier classifier;
      LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn;
      dlib::frontal_face_detector face_detector_hog;
      Utilities::Visualizer visualizer;
    private:
  };

}
