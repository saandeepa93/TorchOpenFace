# **TorchFace**

### This repository is to integrate [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) with PyTorch C++ frontend. Follow the steps below to setup the repo.

# **Installation**
**Follow installation steps in the [wiki](https://github.com/saandeepa93/TorchOpenFace/wiki/Unix-Setup) page.**

# **Usage**
* To load the `libTorchFace.so` library in PyTorch, use the following code.

```
torch.classes.load_library(lib_path)
obj = torch.classes.TorchFaceAnalysis.TorchFace([model_root_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt'])
```

## **FaceLandmarkImg Executable**
```
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


obj = torch.classes.TorchFaceAnalysis.TorchFace([model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt']) # C++ class Object Creation

trans = transforms.ToTensor()
img = Image.open("../data/test_1894_aligned.jpg").convert('RGB')
img = trans(img)
img = img.flip(0) # RGB->BGR. Can do in C++ as well.
img = img.unsqueeze(0).repeat(32, 1, 1, 1) # Sample Batch 
obj.ExtractFeatures(img) # Class Method Call
```

## **Directory Structure**

The primary library which binds OpenFace pure C++ with PyTorch C++ in present under `lib/TorchFace`
The rest of the directories under `lib` are from [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/tree/master/lib/local) original repo. 

```
TorchFace/
├── cmake
├── exe
├── imgs
├── external
└── lib
    ├── CppInerop
    ├── FaceAnalyser
    │   ├── include
    │   └── src
    ├── GazeAnalyser
    │   ├── include
    │   └── src
    ├── LandmarkDetector
    │   ├── include
    │   └── src
    ├── **TorchFace**
    │   ├── include
    │   └── src
    └── Utilities
        ├── include
        └── src

```

## **Sample Output**
<p align="center">
  <img src="./imgs/sample.png" height="290" width="290" >
  <img src="./imgs/sample.jpg" height="290" width="290" >
</p>


