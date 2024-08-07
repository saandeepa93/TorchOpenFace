#include "TorchFaceLandmarkImg.h"
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
TorchFaceLandmarkImg::TorchFaceLandmarkImg(std::vector<std::string> arguments, const c10::Dict<std::string, c10::IValue>& misc_args){

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

void TorchFaceLandmarkImg::SetImageParams(torch::Tensor latest_frame){
  this->image_reader.image_height  = latest_frame.size(1);
  this->image_reader.image_width  = latest_frame.size(2);
  this->image_reader.SetCameraIntrinsics(-1, -1, -1, -1);

}

// Face detect 
std::vector<cv::Rect_<float> > TorchFaceLandmarkImg::FaceDetection(const cv::Mat_<uchar>& grayscale_image, const cv::Mat& rgb_image, \
      const c10::Dict<std::string, c10::IValue>& ex_args, const int& i){
  std::vector<cv::Rect_<float> > face_detections;

  // Add: Include feature to provide bbox values
  if (ex_args.contains("bbox"))
  {
    c10::IValue bboxes = ex_args.at("bbox");
    if (!bboxes.isList()) {
        throw std::runtime_error("IValue is not a list");
    }
    // Iterate over the outer list
    c10::List<c10::IValue> outer_bbox = bboxes.toList();
    c10::IValue bbox = outer_bbox.get(i);

    // Specific BBOX
    std::vector<c10::IValue> elements = bbox.toList().vec();
    float x = elements[0].toDouble();
    float y = elements[1].toDouble();
    float width = elements[2].toDouble() - elements[0].toDouble();
    float height = elements[3].toDouble() - elements[1].toDouble();
    cv::Rect_<float> rect(x, y, width, height);
    
    // Now you can push it back into the vector
    face_detections.push_back(rect);

  }
  // Perform Face Detections
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
      LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, this->face_detector_mtcnn, confidences);
    }
  }
  return face_detections;
}

// FaceLandmarkImg executable code
c10::Dict<std::string, torch::Tensor> TorchFaceLandmarkImg::ExtractFeatures(torch::Tensor rgb_tensors, c10::Dict<std::string, c10::IValue> ex_args){
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
  // Flip channels
  rgb_tensors = rgb_tensors.flip(1);


  // Open Recorder
  this->recording_params = Utilities::RecorderOpenFaceParameters (this->arguments, false, false,
                                                                this->image_reader.fx, this->image_reader.fy, this->image_reader.cx, this->image_reader.cy);

  // IMPORTANT: ASSSUMES ALL IMAGES IN THE BATCH BELONGS TO SAME SUBJECT
  // std::string subject_phase = extract_subject_phase(fnames[0]);
  Utilities::RecorderOpenFace open_face_rec(fnames[0],this->recording_params, this->arguments);

  this->SetImageParams(rgb_tensors[0]);
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

    // Set Camera Params    

    // Step: Perform Face Detection
    std::vector<cv::Rect_<float> > face_detections = this->FaceDetection(grayscale_image, rgb_image, ex_args, i);
    if (face_detections.empty()){
      continue;
    }

    // std::cout<<"Face: "<<i<<"; file: "<<frame_lst[i]<<std::endl;

    // Step: Perform landmark detection for every face detected
    for (size_t face = 0; face < face_detections.size(); ++face)
    {
      // if there are multiple detections go through them
      bool success = LandmarkDetector::DetectLandmarksInImage(rgb_image, face_detections[face], face_model, det_parameters, grayscale_image);

      if (!success)
        continue;

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
      

      // Displaying the tracking visualizations
      // Open Viz    
      if (this->vis){
        this->visualizer.SetImage(rgb_image, this->image_reader.fx, this->image_reader.fy, this->image_reader.cx, this->image_reader.cy);
        this->visualizer.SetObservationFaceAlign(sim_warped_img);
        this->visualizer.SetObservationHOG(hog_descriptor, num_hog_rows, num_hog_cols);
        this->visualizer.SetObservationLandmarks(face_model.detected_landmarks, 1.0, face_model.GetVisibilities()); // Set confidence to high to make sure we always visualize
        this->visualizer.SetObservationPose(pose_estimate, 1.0);
        this->visualizer.SetObservationGaze(gaze_direction0, gaze_direction1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy), face_model.detection_certainty);
        this->visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
      }
      if(this->rec){
        open_face_rec.SetObservationHOG(face_model.detection_success, hog_descriptor, num_hog_rows, num_hog_cols, 31); // The number of channels in HOG is fixed at the moment, as using FHOG
        open_face_rec.SetObservationActionUnits(this->face_analyser.GetCurrentAUsReg(), this->face_analyser.GetCurrentAUsClass());
        open_face_rec.SetObservationLandmarks(face_model.detected_landmarks, face_model.GetShape(image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy),
          face_model.params_global, face_model.params_local, face_model.detection_certainty, face_model.detection_success);
        open_face_rec.SetObservationPose(pose_estimate);
        open_face_rec.SetObservationGaze(gaze_direction0, gaze_direction1, gaze_angle, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy));
        open_face_rec.SetObservationFaceAlign(sim_warped_img);
        open_face_rec.SetObservationFaceID(face);
        open_face_rec.SetObservationFrameName(frame_lst[i]);
        open_face_rec.WriteObservation();
      }


      if (this->vis){
        open_face_rec.SetObservationVisualization(this->visualizer.GetVisImage());
        open_face_rec.WriteObservationTracked();
      }
    }

  }
  
  if (this->rec){
      open_face_rec.Close();
  }

  c10::Dict<std::string, torch::Tensor> all_feats_dict;
  return all_feats_dict;

}



