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

## **Feature Extraction**
```
import torch 
from torchvision import transforms 
from PIL import Image

img = Image.open("../data/sample.png")
img = trans(img)
img = img.flip(0) // Convert to BGR
img = img.unsqueeze(0).repeat(32, 1, 1, 1) // Batch Processing. For now run only for loop

obj.ExtractFeatures(img) //Call Feature Extraction Method
```


