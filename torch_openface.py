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
  "bbox": [204., 136., 326., 283.], 
  "fname": "sample3.png"
}

obj = torch.classes.TorchFaceAnalysis.TorchFace([model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt'])

trans = transforms.ToTensor()
img = Image.open("../data/sample3.png").convert('RGB')
img = trans(img)
img = img.unsqueeze(0).repeat(1, 1, 1, 1) # Sample Batch 


# print(img.min(), img.max())
# img = torch.randn((1, 3, 100, 100))
# min_val = torch.min(img)
# max_val = torch.max(img)
# img = (img - min_val) / (max_val - min_val)

# print(normalized_tensor.size())
# print(normalized_tensor.min(), normalized_tensor.max())
start = time.time()
face_det = obj.ExtractFeatures(img.clone().detach().cpu().contiguous(), misc_args)

# face_det = face_det.permute(0, 2, 3, 1)
# face_det = face_det.flip(0) # RGB->BGR. Can do in C++ as well.
print(face_det.size())
face_det = face_det/255.

save_image(face_det[0], "../data/saved.png")