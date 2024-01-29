#include<iostream>
#include<torch/script.h>
#include "TorchFace.h"
// #include<any>

int main() {
  c10::impl::GenericDict myDict = c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get());
  myDict.insert("key1", c10::IValue("value1"));
  myDict.insert("key2", c10::IValue(42));

  c10::impl::GenericDict newDict;
    for (const auto& kv : dict) {
        // Convert the key to a string, which is a supported type.
        std::string key = kv.key().toStringRef();
        c10::IValue value = kv.value();
        newDict.insert(key, value);
  }


  std::string model_dir = "./data/";
  std::vector<std::string> arguments = {model_dir, "-wild", "-mloc", "./models/model/main_ceclm_general.txt"};

  TorchFaceAnalysis::TorchFace obj(arguments, newDict);

  torch::Tensor img = torch::randn({1, 3, 32, 32});
  // obj.ExtractFeatures(img, myDict);
  std::cout<<"CUDA: "<<torch::cuda::is_available()<<std::endl;
  return 0;
}