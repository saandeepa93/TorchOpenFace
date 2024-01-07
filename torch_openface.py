import torch 
from torchvision import transforms 
from torchvision.utils import save_image
from PIL import Image
import os 
import time

root_dir = "/home/saandeepaath-admin/projects/learning/cpp_cmake/example3"
model_dir = os.path.join(root_dir, "models")
lib_path = os.path.join(root_dir, "./build/lib/TorchFace/libTorchFace.so")
torch.classes.load_library(lib_path)

openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt']
misc_args = {
  'vis': True, 
  'record': True
}
obj = torch.classes.TorchFaceAnalysis.TorchFace([model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt'], misc_args)

trans = transforms.ToTensor()
img = Image.open("../data/sample3.png").convert('RGB')
# img = Image.open("../data/IMG_3110.jpg").convert('RGB')
img = trans(img)
img = img.unsqueeze(0).repeat(1, 1, 1, 1) # Sample Batch 
ex_args = {
  # "bbox": [204., 136., 326., 283.], 
  # "bbox": [734., 450., 1830., 2144.], 
  "vis": True
}

start = time.time()
face_det = obj.ExtractFeatures(img.clone().detach().cpu().contiguous(), ex_args)
print(face_det.size())
face_det = face_det/255.
save_image(face_det[0], "../data/saved.png")