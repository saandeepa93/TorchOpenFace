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


obj = torch.classes.TorchFaceAnalysis.TorchFace([model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt'])

trans = transforms.ToTensor()
img = Image.open("../data/test_1894_aligned.jpg").convert('RGB')
img = trans(img)
img = img.unsqueeze(0).repeat(32, 1, 1, 1) # Sample Batch 

print(img.size())

start = time.time()
face_det = obj.ExtractFeatures(img)

# face_det = face_det.permute(0, 2, 3, 1)
face_det = face_det.flip(0) # RGB->BGR. Can do in C++ as well.
print(face_det.size())
face_det = face_det/255.

save_image(face_det[0], "../data/saved.png")