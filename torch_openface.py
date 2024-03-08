import torch 
from torchvision import transforms 
from torchvision.utils import save_image
from PIL import Image
import os 
import time
from decord import VideoReader
from torchvision.transforms import ToPILImage
from decord import cpu, gpu
from sys import exit as e 
from icecream import ic 
import pandas as pd
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

from transforms import *
from einops import rearrange

print(torch.__version__)
print(torch.cuda.is_available())



class DataAugmentationForVideoMAE(object):
  def __init__(self):
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupScale((224, 224))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        # ADD ANY AUG BEFORE THIS LINE
        ToTorchFormatTensor(div=True),
    ])

  def __call__(self, images):
    process_data = self.transform(images)
    return process_data

  def __repr__(self):
    repr = "(DataAugmentationForVideoMAE,\n"
    repr += "  transform = %s,\n" % str(self.transform)
    repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    repr += ")"
    return repr

def bboxCrop(img_arr, bbox):
  x1, y1, x2, y2 = bbox
  x1 = int(max(0, x1))
  y1 = int(max(0, y1))
  x2 = int(max(0, x2))
  y2 = int(max(0, y2))
  crop_img = img_arr[y1:y2, x1:x2, :]
  crop_img = Image.fromarray(np.array(crop_img)[:, :, ::-1]).convert('RGB')
  return crop_img

def load_pickle(subject, root):
  fpath = os.path.join(root, f"{subject}_rel.pickle")
  with open(fpath, 'rb') as inp:
    pickl_file = pickle.load(inp)
  return pickl_file

root_dir = "/shares/rra_sarkar-2135-1003-00/faces/OpenFace/TorchOpenFace"
model_dir = os.path.join(root_dir, "models")
dest_dir = os.path.join(root_dir, "data/output_dir")
lib_path = os.path.join(root_dir, "./build/lib/TorchFace/libTorchFace.so")

misc_args = {
  "vis": True, 
  "rec": True,
  "first_only": True,
  "gallery_mode": True
}
torch.classes.load_library(lib_path)



def FaceLandmarkImg():
  openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]
  obj = torch.classes.TorchFaceAnalysis.TorchFaceLandmarkImg(openface_args, misc_args)
  print("HERE ORIG")
  trans = transforms.ToTensor()
  img = Image.open(os.path.join(root_dir, "data/sample.jpg")).convert('RGB')
  img = trans(img)
  print(img.size())
  img = img.unsqueeze(0).repeat(2, 1, 1, 1) # Sample Batch 
  frame_lst = ["f1.jpg", "f2.jpg"]

  # fnames_lst = [os.path.join(dest_dir, k) for k in fnames]
  ex_args = {
    # "bbox": [[204., 136., 326., 283.], [204., 136., 326., 288.]], 
    # "bbox": [734., 450., 1830., 2144.], 
    "fname": ["bgc1/sub1"],
    "frame_lst": frame_lst
  }

  start = time.time()
  features = obj.ExtractFeatures(img.clone().detach().cpu().contiguous(), ex_args)
  for keys, values in features.items():
    print(f"{keys}: {values.size()}")
    

def FeatureExtraction():
  openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]
  obj = torch.classes.TorchFaceAnalysis.TorchFeatureExtraction(openface_args, misc_args)
  
  trans = DataAugmentationForVideoMAE()
  
  vid_path = os.path.join(root_dir, "data/G00355_set1_rand_1622054879793_c54ddb3b.mp4")
  save = "/shares/rra_sarkar-2135-1003-00/faces/BGC_pickles_v4/"
  subject_phase = "G00355_BGC1"
  subject_dict = load_pickle(f"{subject_phase}", save)
  
  all_vid = subject_dict['field']
  for i in range(len(all_vid['fpath'])):
    print(all_vid['fpath'][i])
    if "G00355_set1_rand_1622054879793_c54ddb3b.mp4" in all_vid['fpath'][i]:
      bbox = all_vid['bbox'][i]
      frame_lst = all_vid['frame_num'][i]
      break
  

  vr = VideoReader(vid_path, ctx=cpu(0))

  # SORT
  zipped_lists = zip(frame_lst, bbox)
  sorted_pairs = sorted(zipped_lists)
  # frame_lst_all, bbox_all = zip(*sorted_pairs)
  frame_lst, bbox = zip(*sorted_pairs)

  # SAMPLING
  # nframe_ind = random.sample(range(len(frame_lst)), 400)

  # for frame_num in range(0, len(frame_lst_all), 200):

    # print(f"Processing Frames range : {frame_num, frame_num+200}")
    # frame_lst = frame_lst_all[frame_num:frame_num+200]
    # bbox = bbox_all[frame_num:frame_num+200]

    # nframe_ind = frame_lst
  nframe_ind = list(range(len(frame_lst)))
  nframe_lst = [frame_lst[i] for i in nframe_ind]
  bbox_lst = [bbox[i] for i in nframe_ind]

  # LOAD FRAME
  frames = vr.get_batch(nframe_lst).asnumpy()

  # CROP FACE
  cropped_images = []
  for img_arr, bbox in zip(frames, bbox_lst):
    crop_img = bboxCrop(img_arr, bbox)
    cropped_images.append(crop_img)


  frames = trans(cropped_images)
  video = frames.contiguous().view((len(nframe_lst), 3) + frames.size()[-2:]).transpose(0,1)
  # imgs = [ ToPILImage()(video[:, vid, :, :].cpu().clamp(0,0.996)) for vid in range(video.shape[1])  ]
  # for id, im in enumerate(imgs):
  #   im.save(f"./data/loaders/{id}.jpg")
  
  ex_args = {
    "fname": ["G00355_set1_rand_1622054879793_c54ddb3b"],
    'frame_lst': [str(i) for i in nframe_lst]
  }
  
  start = time.time()
  video = rearrange(video, 'c t h w -> t c h w')
  ic(video.size(), video.min(), video.max())
  obj.ExtractFeatures(video.clone().detach().cpu().contiguous(), ex_args)
    


if __name__ == "__main__":
  FeatureExtraction()
  # FaceLandmarkImg()

  # csv_path = os.path.join(root_dir, "data/output_dir/bgc1/sub1/sub1.csv")
  # df = pd.read_csv(csv_path)
  # print(df.head())