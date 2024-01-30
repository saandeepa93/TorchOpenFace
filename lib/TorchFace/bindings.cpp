#include <torch/script.h>
#include <torch/torch.h>

#include "TorchFace.h"


using namespace TorchFaceAnalysis;


/***************************C++ -> Torch Bindings***********************************/
TORCH_LIBRARY(TorchFaceAnalysis, m) {
  m.class_<TorchFaceAnalysis::TorchFace>("TorchFace")
    .def(torch::init<std::vector<std::string>, const c10::Dict<std::string, c10::IValue>&>())
    .def("ExtractFeatures", &TorchFace::ExtractFeatures)
    ;
}