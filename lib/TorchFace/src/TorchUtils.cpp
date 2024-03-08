
#include<iostream>
#include "TorchUtils.h"

namespace TorchUtilities{

  /***************************Function Overloading***********************************/
  torch::Tensor ToTemplateTensor(const cv::Mat& mat){
    torch::Tensor tensor = torch::from_blob(mat.data, { mat.rows, mat.cols, 3 }, at::kByte);
    return tensor.clone();
  }

    torch::Tensor ToTemplateTensor(const cv::Mat_<float>& mat){
      torch::Tensor tensor = torch::from_blob(mat.data, {mat.rows, mat.cols}, torch::kFloat32);
      return tensor.clone();
    }


  torch::Tensor ToTemplateTensor(const std::vector<double>& mat){
    torch::Tensor tensor = torch::tensor(std::vector<double>(mat.begin(), mat.end()), torch::kDouble);
    return tensor.clone();
  }

  torch::Tensor ToTemplateTensor(const cv::Vec6d& mat){
    torch::Tensor tensor = torch::from_blob(const_cast<double*>(mat.val), {6}, torch::kDouble);
    return tensor.clone();
  }



  /***************************Utility Functions***********************************/
  std::vector<double> ToVector(const cv::Rect_<float>& rect) {
      std::vector<double> vec;
      vec.push_back(static_cast<double>(rect.x));
      vec.push_back(static_cast<double>(rect.y));
      vec.push_back(static_cast<double>(rect.width));
      vec.push_back(static_cast<double>(rect.height));
      return vec;
  }


  std::vector<std::string> ToVectString(c10::IValue ivalues){
    std::vector<std::string> strings;
    // Get the list of IValues
    c10::List<c10::IValue> list = ivalues.toList();
    // Iterate over the list and convert each IValue to a string
    for (const c10::IValue& item : list) {
      strings.push_back(item.toStringRef());
    }

    return strings;
  }

  cv::Mat ToMat(torch::Tensor rgb_tensor){
    rgb_tensor = rgb_tensor.permute({1, 2, 0}).contiguous();
    rgb_tensor = rgb_tensor.mul(255).clamp(0, 255).to(torch::kU8);
    cv::Mat rgb_mat = cv::Mat(rgb_tensor.size(0), rgb_tensor.size(1), CV_8UC3, rgb_tensor.data_ptr());
    return rgb_mat.clone();

  }

  /***************************Simple Helpers***********************************/

  void print_vec(std::vector<std::string> args){
    for(const std::string& str : args){
      // std::cout<<"Args: "<<str<<std::endl;
    }
  }

  void print_au(std::vector<std::pair<std::string, double>> data){
    for (const auto& pair : data) {
          std::cout << pair.first << ": " << pair.second << std::endl;
      }
  }

  void write_cv_img(cv::Mat img, std::string fname){
    cv::imwrite("../data/" + fname + ".png", img);
  }


  std::string extract_subject_phase(std::string str){
    size_t pos = str.find_last_of('/'); // find position of last '/'
    std::string sub = str.substr(0, pos); // extract substring
    return sub;

  }


}