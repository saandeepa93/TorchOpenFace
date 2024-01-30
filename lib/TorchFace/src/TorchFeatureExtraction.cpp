#include "TorchFace.h"
#include <opencv2/dnn.hpp>
#include <typeinfo>
#include "TorchUtils.h"


using namespace TorchFaceAnalysis;
using namespace TorchUtilities;



TorchVideoFace::TorchVideoFace(std::vector<std::string> arguments, const c10::Dict<std::string, c10::IValue>& misc_args){
  this->arguments = arguments;
  this->vis = misc_args.at("vis").toBool();
  this->rec = misc_args.at("rec").toBool();
  this->first_only = misc_args.at("first_only").toBool();


  // Set parameters for Face detection models
	this->det_parameters = LandmarkDetector::FaceModelParameters (arguments);
  // Set parameters for Facial Analysis module (AU, Gaze, Headpose etc..)
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
  face_analysis_params.OptimizeForImages();
  // load a face detectors
  this->classifier = cv::CascadeClassifier(this->det_parameters.haar_face_detector_location);
	this->face_detector_hog = dlib::get_frontal_face_detector();
	this->face_detector_mtcnn = LandmarkDetector::FaceDetectorMTCNN (this->det_parameters.mtcnn_face_detector_location);
	// load LM model
  this->face_model = LandmarkDetector::CLNF(det_parameters.model_location);
  if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
	}
  // Load Face Analyser (AU)
	this->face_analyser = FaceAnalysis::FaceAnalyser(face_analysis_params);
  // A utility for visualizing the results
  if(this->vis){
    this->visualizer = Utilities::Visualizer (arguments);
  }
  
}