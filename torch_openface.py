import torch 
from torchvision import transforms 
from torchvision.utils import save_image
from PIL import Image
import os 
import time

root_dir = "/home/saandeepaath-admin/projects/learning/cpp_cmake/example3"
model_dir = os.path.join(root_dir, "models")
dest_dir = os.path.join(root_dir, "data/output_dir")
lib_path = os.path.join(root_dir, "./build/lib/TorchFace/libTorchFace.so")

openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]
misc_args = {
  "vis": True, 
  "rec": True,
  "first_only": True
}
torch.classes.load_library(lib_path)
obj = torch.classes.TorchFaceAnalysis.TorchFace(openface_args, misc_args)

trans = transforms.ToTensor()
img = Image.open("../data/sample3.png").convert('RGB')
# img = Image.open("../data/IMG_3110.jpg").convert('RGB')
img = trans(img)
img = img.unsqueeze(0).repeat(2, 1, 1, 1) # Sample Batch 
ex_args = {
  "bbox": [[204., 136., 326., 283.], [204., 136., 326., 288.]], 
  # "bbox": [734., 450., 1830., 2144.], 
  "fname": ["saandeep2", "saande3"]
}

start = time.time()
features = obj.ExtractFeatures(img.clone().detach().cpu().contiguous(), ex_args)
for keys, values in features.items():
  print(f"{keys}: {values.size()}")