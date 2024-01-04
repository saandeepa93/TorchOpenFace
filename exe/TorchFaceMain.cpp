#include<iostream>
#include<torch/torch.h>
#include "TorchFace.h"

int main() {
  
  TorchFaceAnalysis::TorchFace obj = TorchFaceAnalysis::TorchFace({"test", "Test2"});

  std::cout<<"CUDA: "<<torch::cuda::is_available()<<std::endl;
  return 0;
}