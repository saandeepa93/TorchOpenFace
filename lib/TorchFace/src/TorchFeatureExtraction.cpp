#include "TorchFeatureExtraction.h"
#include <opencv2/dnn.hpp>
#include <typeinfo>
#include "TorchUtils.h"


using namespace TorchFaceAnalysis;
using namespace TorchUtilities;





/***************************Template Functions***********************************/
// BUG: TorchLib does not support compiled template function headers in .so. Needs to be statically linked.
template <typename T>
std::vector<T> squeeze(const std::vector<std::vector<T>>& vecVec) {
    std::vector<T> vec;
    for (const auto& innerVec : vecVec) {
        T element = innerVec[0];
        vec.push_back(element);
    }
    return vec;
}

template <typename T>
torch::Tensor ToTensor(std::vector<T> mats, std::string template_type){
  std::vector<torch::Tensor> batched_warped_tensors;
  for(const T& mat: mats){
    torch::Tensor tensor = ToTemplateTensor(mat);
    if(template_type == "Mat"){
      tensor = tensor.to(at::kFloat);
      tensor = tensor.permute({2, 0, 1});
    }
    batched_warped_tensors.push_back(tensor);
  }
  torch::Tensor batchedTensor = torch::stack(batched_warped_tensors);
  if(template_type == "Mat")
    batchedTensor = batchedTensor.flip(1);
  return batchedTensor;
}


/***************************Class Methods***********************************/

// Class primary methods
// Constructor
TorchFeatureExtraction::TorchFeatureExtraction(std::vector<std::string> arguments, const c10::Dict<std::string, c10::IValue>& misc_args){

  this->arguments = arguments;
  this->vis = misc_args.at("vis").toBool();
  this->rec = misc_args.at("rec").toBool();
  this->first_only = misc_args.at("first_only").toBool();
  this->gallery_mode = misc_args.at("gallery_mode").toBool();


  // Set parameters for Face detection models
	this->det_parameters = LandmarkDetector::FaceModelParameters (arguments);
  // Set parameters for Facial Analysis module (AU, Gaze, Headpose etc..)
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
  face_analysis_params.OptimizeForImages();
	
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

void TorchFeatureExtraction::SetImageParams(torch::Tensor latest_frame){
  this->sequence_reader.frame_height = latest_frame.size(1);
  this->sequence_reader.frame_width = latest_frame.size(2);
  this->sequence_reader.SetCameraIntrinsics(-1, -1, -1, -1);

}


// FaceLandmarkImg executable code
c10::Dict<std::string, torch::Tensor> TorchFeatureExtraction::ExtractFeatures(torch::Tensor rgb_tensors, c10::Dict<std::string, c10::IValue> ex_args){
  // Variables to hold data per image
  std::vector<std::vector<cv::Mat>> batch_sim_warped_img;
  std::vector<std::vector<std::vector<double>>> batch_face_detection;
  std::vector<std::vector<std::vector<double>>> batch_au_intensities;
  std::vector<std::vector<std::vector<double>>> batch_au_occurence;
  std::vector<std::vector<cv::Mat_<float>>> batch_detected_landmarks;
  std::vector<std::vector<cv::Vec6d>> batch_head_pose;
  std::vector<std::vector<bool>> success_flags;
  
  // Get all fnames
  std::vector<std::string> fnames = ToVectString(ex_args.at("fname"));

  std::vector<std::string> frame_lst = ToVectString(ex_args.at("frame_lst"));
  // if (this->gallery_mode)
  //   std::vector<std::string> frame_lst = ToVectString(ex_args.at("frame_lst"));
  // else
  //   std::vector<int64_t> frame_lst = ex_args.at("frame_lst").toIntVector();

  // Flip channels
  rgb_tensors = rgb_tensors.flip(1);

  // Open Recorder
  this->recording_params = Utilities::RecorderOpenFaceParameters (this->arguments, true, false,
                                                                this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy);
  Utilities::RecorderOpenFace open_face_rec(fnames[0],this->recording_params, this->arguments);

  // Set Camera Params    
  this->SetImageParams(rgb_tensors[0]);
  this->fps_tracker.AddFrame();
  for (int i = 0; i < rgb_tensors.size(0); ++i){

    // Variables to hold data per face in an image
    std::vector<cv::Mat> single_sim_warped_img;
    std::vector<std::vector<double>> single_face_detection;
    std::vector<std::vector<double>> single_au_intensities;
    std::vector<std::vector<double>> single_au_occurence;
    std::vector<cv::Mat_<float>> single_detected_landmarks;
    std::vector<cv::Vec6d> single_head_pose;
    std::vector<bool> success_flags;
    

    cv::Mat rgb_image = ToMat(rgb_tensors[i]);
    cv::Mat_<uchar> grayscale_image;
    Utilities::ConvertToGrayscale_8bit(rgb_image, grayscale_image);

    // Step: Perform landmark detection for every face detected
    bool success = LandmarkDetector::DetectLandmarksInVideo(rgb_image,  this->face_model, det_parameters, grayscale_image);

    if (!success){
      continue;
    }

    // Estimate head pose and eye gaze				
    cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
    
    // Gaze tracking, absolute gaze direction
    cv::Point3f gazeDirection0(0, 0, 0); 
    cv::Point3f gazeDirection1(0, 0, 0); 
    cv::Vec2d gazeAngle(0, 0);
    if (success && face_model.eye_model)
    {
      GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
      GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
      gazeAngle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
    }

    // Perform AU detection and HOG feature extraction, as this can be expensive only compute it if needed by output or visualization
    cv::Mat sim_warped_img;
    cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;
    this->face_analyser.PredictStaticAUsAndComputeFeatures(rgb_image, face_model.detected_landmarks);
    this->face_analyser.GetLatestAlignedFace(sim_warped_img);
    this->face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
    
    // Keeping track of FPS
    this->fps_tracker.AddFrame();
    // Displaying the tracking visualizations
    // Open Viz    
    if (this->vis){
      this->visualizer.SetImage(rgb_image, this->sequence_reader.fx, this->sequence_reader.fy, this->sequence_reader.cx, this->sequence_reader.cy);
      this->visualizer.SetObservationFaceAlign(sim_warped_img);
      this->visualizer.SetObservationHOG(hog_descriptor, num_hog_rows, num_hog_cols);
      this->visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities()); // Set confidence to high to make sure we always visualize
      this->visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
      this->visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
      this->visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
      visualizer.SetFps(this->fps_tracker.GetFPS());
    }
    if(this->rec){
      open_face_rec.SetObservationHOG(success, hog_descriptor, num_hog_rows, num_hog_cols, 31); // The number of channels in HOG is fixed at the moment, as using FHOG
      open_face_rec.SetObservationActionUnits(this->face_analyser.GetCurrentAUsReg(), this->face_analyser.GetCurrentAUsClass());
      open_face_rec.SetObservationLandmarks(face_model.detected_landmarks, face_model.GetShape(sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy),
        face_model.params_global, face_model.params_local, face_model.detection_certainty, success);
      open_face_rec.SetObservationPose(pose_estimate);
      open_face_rec.SetObservationGaze(gazeDirection0, gazeDirection1, gazeAngle, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy));
      open_face_rec.SetObservationFaceAlign(sim_warped_img);
      open_face_rec.SetObservationFaceID(0);
      // open_face_rec.SetObservationFrameNumber(frame_lst[i]);
      open_face_rec.SetObservationFrameNumber(i);
      if (this->gallery_mode){
        open_face_rec.SetObservationFrameName(frame_lst[i]);
      }
      open_face_rec.WriteObservation();
    }


    if (this->vis){
      open_face_rec.SetObservationVisualization(this->visualizer.GetVisImage());
      open_face_rec.WriteObservationTracked();
    }


  }
  
  if (this->rec){
      open_face_rec.Close();
      sequence_reader.Close();
  }

  c10::Dict<std::string, torch::Tensor> all_feats_dict;
  return all_feats_dict;

}



