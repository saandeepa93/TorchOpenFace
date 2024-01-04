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
# img = Image.open("../data/sample.png")
img = trans(img)

img = img.flip(0)
img = img.unsqueeze(0).repeat(32, 1, 1, 1)

print(img.size())

start = time.time()
aus = obj.ExtractFeatures(img)
# save_image(aus, '../data/img1.png')
print(f"Total: {time.time() - start}")
print(aus.size())
print(type(aus))