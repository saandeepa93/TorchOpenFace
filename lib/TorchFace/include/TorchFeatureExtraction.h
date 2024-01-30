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

#ifndef TORCHFACE_H
#define TORCHFACE_H

namespace TorchFaceAnalysis{

  class TorchVideoFace:public torch::CustomClassHolder {
    public:
      TorchVideoFace(std::vector<std::string> arguments, const c10::Dict<std::string, c10::IValue>& misc_args);
      c10::Dict<std::string, torch::Tensor> ExtractFeatures(torch::Tensor img, c10::Dict<std::string, c10::IValue> ex_args);
      void SetImageParams(cv::Mat img);
      std::vector<cv::Rect_<float> > FaceDetection(const cv::Mat_<uchar>& grayscale_image, const cv::Mat& rgb_image, \
          const c10::Dict<std::string, c10::IValue>& ex_args, const int& i);

      std::vector<std::string> arguments;
      bool vis; bool rec; bool first_only;
      
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
      //ERROR: unable to declare c10::Dict<std::string, c10::IValue> type. Need appropriate constructor initialization
      // c10::Dict<std::string, c10::IValue> misc_args = c10::impl::GenericDict<std::string, c10::IValue>();
      
  };

}

#endif
