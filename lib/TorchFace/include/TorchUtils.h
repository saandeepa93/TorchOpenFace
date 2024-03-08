#pragma once
#include<iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <map>
#include <any>
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include<vector>
#include <opencv2/imgcodecs.hpp>
#include "ImageManipulationHelpers.h"

namespace TorchUtilities{

   /***************************Function Overloading***********************************/
  torch::Tensor ToTemplateTensor(const cv::Mat& mat);
  torch::Tensor ToTemplateTensor(const cv::Mat_<float>& mat);
  torch::Tensor ToTemplateTensor(const std::vector<double>& mat);
  torch::Tensor ToTemplateTensor(const cv::Vec6d& mat);

  // /***************************Utility Functions***********************************/
  std::vector<double> ToVector(const cv::Rect_<float>& rect);
  std::vector<std::string> ToVectString(c10::IValue ivalues);
  cv::Mat ToMat(torch::Tensor rgb_tensor);

  void print_vec(std::vector<std::string> args);
  void print_au(std::vector<std::pair<std::string, double>> data);
  void write_cv_img(cv::Mat img, std::string fname);
  std::string extract_subject_phase(std::string str);

};