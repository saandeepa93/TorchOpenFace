#include "TorchFace.h"

using namespace TorchFaceAnalysis;

// C++ -> Torch Bindings
TORCH_LIBRARY(TorchFaceAnalysis, m) {
  m.class_<TorchFaceAnalysis::TorchFace>("TorchFace")
    .def(torch::init<std::vector<std::string>>())
    .def("getter", &TorchFace::getter)
    .def("setter", &TorchFace::setter)
    ;
  
}

void print_vec(std::vector<std::string> args){
  for(const std::string& str : args){
    std::cout<<"Args: "<<str<<std::endl;
  }
}

// Constructor
TorchFace::TorchFace(std::vector<std::string> arguments) {
  print_vec(arguments);

  // Load the models if images found
	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// The modules that are being used for tracking
	std::cout << "Loading the model" << std::endl;
  // LandmarkDetector::CLNF face_model(det_parameters.model_location);
  this->face_model = LandmarkDetector::CLNF(det_parameters.model_location);
  
  if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
	}

	std::cout << "Model loaded" << std::endl;
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);

  face_analysis_params.OptimizeForImages();
	// FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);
	this->face_analyser = FaceAnalysis::FaceAnalyser(face_analysis_params);

  // If bounding boxes not provided, use a face detector
	// cv::CascadeClassifier classifier(det_parameters.haar_face_detector_location);
	// dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();

	// LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn(det_parameters.mtcnn_face_detector_location);
	this->face_detector_mtcnn = LandmarkDetector::FaceDetectorMTCNN (det_parameters.mtcnn_face_detector_location);

}


std::string TorchFace::getter(){
  return this->model_root;
}

void TorchFace::setter(std::string root){
  this->model_root = root;

}

