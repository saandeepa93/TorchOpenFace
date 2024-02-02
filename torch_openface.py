import torch 
from torchvision import transforms 
from torchvision.utils import save_image
from PIL import Image
import os 
import time
from decord import VideoReader
from decord import cpu, gpu
from sys import exit as e 
from icecream import ic 
import random
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

root_dir = "/home/saandeepaath-admin/projects/learning/cpp_cmake/example3"
model_dir = os.path.join(root_dir, "models")
dest_dir = os.path.join(root_dir, "data/output_dir")
lib_path = os.path.join(root_dir, "./build/lib/TorchFace/libTorchFace.so")

misc_args = {
  "vis": True, 
  "rec": True,
  "first_only": True
}
torch.classes.load_library(lib_path)

def FaceLandmarkImg():
  openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]
  obj = torch.classes.TorchFaceAnalysis.TorchFaceLandmarkImg(openface_args, misc_args)
  trans = transforms.ToTensor()
  img = Image.open("/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/data/sample3.png").convert('RGB')
  img = trans(img)
  img = img.unsqueeze(0).repeat(2, 1, 1, 1) # Sample Batch 
  ex_args = {
    "bbox": [[204., 136., 326., 283.], [204., 136., 326., 288.]], 
    # "bbox": [734., 450., 1830., 2144.], 
    "fname": ["video_ouptut"]
  }

  start = time.time()
  features = obj.ExtractFeatures(img.clone().detach().cpu().contiguous(), ex_args)
  for keys, values in features.items():
    print(f"{keys}: {values.size()}")
    

def FeatureExtraction():
  # openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]
  openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]

  obj = torch.classes.TorchFaceAnalysis.TorchFeatureExtraction(openface_args, misc_args)
  trans = transforms.ToTensor()
  
  vid_path = os.path.join(root_dir, "data/video01.mp4")
  vr = VideoReader(vid_path, ctx=cpu(0))
  
  frame_lst = random.sample(list(range(0, len(vr))), 50)
  # frame_lst = random.sample(list(range(0, len(vr))), len(vr))
  # frame_lst.sort()
  frames = vr.get_batch(frame_lst)
  frames = frames.asnumpy()
  video = torch.from_numpy(frames)
  video = video.permute(0, 3, 1, 2)
  video = torch.div(video, 255.)
  print(video.size(), video.dtype)
  
  ex_args = {
    # "bbox": [[204., 136., 326., 283.], [204., 136., 326., 288.]], 
    # "bbox": [734., 450., 1830., 2144.], 
    "fname": ["video01"],
    'frame_lst': frame_lst
  }
  
  print(len(frame_lst), video.size())

  start = time.time()
  features = obj.ExtractFeatures(video.clone().detach().cpu().contiguous(), ex_args)
  # for keys, values in features.items():
  #   print(f"{keys}: {values.size()}")
    


if __name__ == "__main__":
  FeatureExtraction()
  # FaceLandmarkImg()