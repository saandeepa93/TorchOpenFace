#include "TorchFace.h"
#include <opencv2/dnn.hpp>


using namespace TorchFaceAnalysis;

// C++ -> Torch Bindings
TORCH_LIBRARY(TorchFaceAnalysis, m) {
  m.class_<TorchFaceAnalysis::TorchFace>("TorchFace")
    .def(torch::init<std::vector<std::string>>())
    .def("ExtractFeatures", &TorchFace::ExtractFeatures)
    ;
  
}


// Helper functions
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

cv::Mat TensorToMat(torch::Tensor rgb_tensor){
  // contiguos is IMPORTANT 
  rgb_tensor = rgb_tensor.permute({1, 2, 0}).contiguous();
  // Scale the tensor values to the range [0, 255] and convert it to 8-bit unsigned integer
  rgb_tensor = rgb_tensor.mul(255).clamp(0, 255).to(torch::kU8);
  // Actual Conversion
  cv::Mat rgb_mat = cv::Mat(rgb_tensor.size(0), rgb_tensor.size(1), CV_8UC3, rgb_tensor.data_ptr());
  // Return Clone
  return rgb_mat;

}

torch::Tensor MatToTensor(std::vector<cv::Mat> mats){
  std::vector<torch::Tensor> batched_warped_tensors;
  for(const cv::Mat& mat: mats){
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    cv::Mat matFloat;
    mat.convertTo(matFloat, CV_32F, 1.0 / 255);
    auto size = matFloat.size();
    auto nChannels = matFloat.channels();
    auto tensor = torch::from_blob(matFloat.data, {size.height, size.width, nChannels});
    batched_warped_tensors.push_back(tensor.permute({2, 0, 1}));
  }
  torch::Tensor batchedTensor = torch::stack(batched_warped_tensors);
  return batchedTensor;
}



// Class helper methods
void TorchFace::SetImageParams(cv::Mat latest_frame){
  this->image_reader.image_height = latest_frame.size().height;
  this->image_reader.image_width = latest_frame.size().width;
  this->image_reader.SetCameraIntrinsics(-1, -1, -1, -1);

}

// Class primary methods
// Constructor
TorchFace::TorchFace(std::vector<std::string> arguments) {
  // Set parameters for Face detection models
  this->arguments = arguments;
	this->det_parameters = LandmarkDetector::FaceModelParameters (arguments);

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

  // Set parameters for Facial Analysis module (AU, Gaze, Headpose etc..)
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
  face_analysis_params.OptimizeForImages();
	this->face_analyser = FaceAnalysis::FaceAnalyser(face_analysis_params);

  // A utility for visualizing the results
	this->visualizer = Utilities::Visualizer (arguments);
  
  
}

// Face detect and LM
std::vector<cv::Rect_<float> >  TorchFace::FaceDetectionAndLM(const cv::Mat_<uchar>& grayscale_image, const cv::Mat& rgb_image){
  // Step: Perform Face Detection
  std::vector<cv::Rect_<float> > face_detections;
  // Add: Include feature to provide bbox values
  if (has_bounding_boxes)
  {
  }
  else
  {
    if (this->det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
    {
      std::vector<float> confidences;
      LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);
    }
    else if (this->det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HAAR_DETECTOR)
    {
      LandmarkDetector::DetectFaces(face_detections, grayscale_image, this->classifier);
    }
    else
    {
      std::vector<float> confidences;
      // std::cout<<"IMAGE SIZE: "<<rgb_image.size()<<std::endl;
      LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, this->face_detector_mtcnn, confidences);
    }
  }
  return face_detections;
}

// FaceLandmarkImg executable code
torch::Tensor TorchFace::ExtractFeatures(torch::Tensor rgb_tensors){

  std::vector<cv::Mat> batch_sim_warped_img;

  for (int i = 0; i < rgb_tensors.size(0); ++i){
    cv::Mat rgb_image = TensorToMat(rgb_tensors[i]);
    cv::Mat_<uchar> grayscale_image;
    Utilities::ConvertToGrayscale_8bit(rgb_image, grayscale_image);

    // Set Camera Params    
    this->SetImageParams(rgb_image);
    // Open Recorder
    this->recording_params = Utilities::RecorderOpenFaceParameters (this->arguments, false, false,
                                                                  this->image_reader.fx, this->image_reader.fy, this->image_reader.cx, this->image_reader.cy);
    Utilities::RecorderOpenFace open_face_rec("sample",this->recording_params, this->arguments);
    // Open Viz    
    this->visualizer.SetImage(rgb_image, this->image_reader.fx, this->image_reader.fy, this->image_reader.cx, this->image_reader.cy);
    

    // Step: Perform Face Detection
    std::vector<cv::Rect_<float> > face_detections = this->FaceDetectionAndLM(grayscale_image, rgb_image);

    // // Add: Include feature to provide bbox values
    // if (has_bounding_boxes)
    // {
    // }
    // else
    // {
    //   if (this->det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
    //   {
    //     std::vector<float> confidences;
    //     LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);
    //   }
    //   else if (this->det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HAAR_DETECTOR)
    //   {
    //     LandmarkDetector::DetectFaces(face_detections, grayscale_image, this->classifier);
    //   }
    //   else
    //   {
    //     std::vector<float> confidences;
    //     // std::cout<<"IMAGE SIZE: "<<rgb_image.size()<<std::endl;
    //     LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, this->face_detector_mtcnn, confidences);
    //   }
    // }


    // Step: Perform landmark detection for every face detected
    int face_det = 0;
    for (size_t face = 0; face < face_detections.size(); ++face)
    {
      // if there are multiple detections go through them
      bool success = LandmarkDetector::DetectLandmarksInImage(rgb_image, face_detections[face], face_model, det_parameters, grayscale_image);

      // Estimate head pose and eye gaze				
      cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy);
      
      // Gaze tracking, absolute gaze direction
      cv::Point3f gaze_direction0(0, 0, -1);
      cv::Point3f gaze_direction1(0, 0, -1);
      cv::Vec2f gaze_angle(0, 0);
      if (face_model.eye_model)
      {
        GazeAnalysis::EstimateGaze(face_model, gaze_direction0, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy, true);
        GazeAnalysis::EstimateGaze(face_model, gaze_direction1, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy, false);
        gaze_angle = GazeAnalysis::GetGazeAngle(gaze_direction0, gaze_direction1);
      }

      // Perform AU detection and HOG feature extraction, as this can be expensive only compute it if needed by output or visualization
      cv::Mat sim_warped_img;
      cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;
      this->face_analyser.PredictStaticAUsAndComputeFeatures(rgb_image, face_model.detected_landmarks);
      this->face_analyser.GetLatestAlignedFace(sim_warped_img);
      this->face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
      batch_sim_warped_img.push_back(sim_warped_img);

      // Displaying the tracking visualizations
			this->visualizer.SetObservationFaceAlign(sim_warped_img);
			this->visualizer.SetObservationHOG(hog_descriptor, num_hog_rows, num_hog_cols);
			this->visualizer.SetObservationLandmarks(face_model.detected_landmarks, 1.0, face_model.GetVisibilities()); // Set confidence to high to make sure we always visualize
			this->visualizer.SetObservationPose(pose_estimate, 1.0);
			this->visualizer.SetObservationGaze(gaze_direction0, gaze_direction1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy), face_model.detection_certainty);
			this->visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
      // print_au(this->face_analyser.GetCurrentAUsReg());

      // write_cv_img(sim_warped_img, "warped");
      open_face_rec.SetObservationHOG(face_model.detection_success, hog_descriptor, num_hog_rows, num_hog_cols, 31); // The number of channels in HOG is fixed at the moment, as using FHOG
			open_face_rec.SetObservationActionUnits(this->face_analyser.GetCurrentAUsReg(), this->face_analyser.GetCurrentAUsClass());
			open_face_rec.SetObservationLandmarks(face_model.detected_landmarks, face_model.GetShape(image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy),
				face_model.params_global, face_model.params_local, face_model.detection_certainty, face_model.detection_success);
			open_face_rec.SetObservationPose(pose_estimate);
			open_face_rec.SetObservationGaze(gaze_direction0, gaze_direction1, gaze_angle, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy));
			open_face_rec.SetObservationFaceAlign(sim_warped_img);
			open_face_rec.SetObservationFaceID(face);
			open_face_rec.WriteObservation();


    }

    open_face_rec.SetObservationVisualization(this->visualizer.GetVisImage());
		open_face_rec.WriteObservationTracked();

		open_face_rec.Close();


  }

  torch::Tensor warped_tensor = MatToTensor(batch_sim_warped_img);
  return warped_tensor;

}



