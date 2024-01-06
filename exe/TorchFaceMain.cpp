#include<iostream>
// #include<torch/torch.h>
#include "TorchFace.h"
// #include<any>

int main() {
    // c10::impl::GenericDict myDict(c10::StringType::get(), c10::IValueType::get());
    c10::Dict<std::string, c10::IValue> myDict;

    // Insert an integer
    myDict.insert("frame", c10::IValue(1));

    // Insert a string
    myDict.insert("second", c10::IValue("string"));  


  std::string model_dir = "/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/models";
  TorchFaceAnalysis::TorchFace obj({model_dir, "-wild", "-mloc", "./models/model/main_ceclm_general.txt"});

  torch::Tensor img = torch::randn({1, 3, 32, 32});
  // obj.ExtractFeatures(img, myDict);
  std::cout<<"CUDA: "<<torch::cuda::is_available()<<std::endl;
  return 0;
}