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

### **Individual images**
```
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
img = trans(img)
img = img.unsqueeze(0).repeat(2, 1, 1, 1) # Sample Batch 
ex_args = {
  "bbox": [[204., 136., 326., 283.], [204., 136., 326., 288.]], 
  "fname": ["saandeep2", "saande3"]
}

start = time.time()
features = obj.ExtractFeatures(img.clone().detach().cpu().contiguous(), ex_args)
for keys, values in features.items():
  print(f"{keys}: {values.size()}")
```

### **Using DataLoader**
```
  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  
  torch.classes.load_library("<dir>/libTorchFace.so")
  obj = torch.classes.TorchFaceAnalysis.TorchFace([model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt'])
  
  test_set = RafDb(cfg, "val", transform, transform)
  test_loader = DataLoader(test_set, batch_size=cfg.TRAINING.BATCH, shuffle=False, num_workers = cfg.DATASET.NUM_WORKERS)
  
  # BBOX (0, 0, max, max) since RAF-DB images are already aligned!
  misc_args = {
    "bbox":[0., 0., float(cfg.DATASET.IMG_SIZE), float(cfg.DATASET.IMG_SIZE)]
  }

  for b, (img, _) in enumerate(tqdm(test_loader)):
    obj.ExtractFeatures(img.cpu().detach(), misc_args)
    e()
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


