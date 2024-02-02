#include <torch/script.h>
#include <torch/torch.h>

#include "TorchFaceLandmarkImg.h"
#include "TorchFeatureExtraction.h"


using namespace TorchFaceAnalysis;


/***************************C++ -> Torch Bindings***********************************/
TORCH_LIBRARY(TorchFaceAnalysis, m) {
  m.class_<TorchFaceAnalysis::TorchFaceLandmarkImg>("TorchFaceLandmarkImg")
    .def(torch::init<std::vector<std::string>, const c10::Dict<std::string, c10::IValue>&>())
    .def("ExtractFeatures", &TorchFaceLandmarkImg::ExtractFeatures)
    ;

  m.class_<TorchFaceAnalysis::TorchFeatureExtraction>("TorchFeatureExtraction")
    .def(torch::init<std::vector<std::string>, const c10::Dict<std::string, c10::IValue>&>())
    .def("ExtractFeatures", &TorchFeatureExtraction::ExtractFeatures)
    ;
}