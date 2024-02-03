# **TorchFace**

### This repository is to compile [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) with PyTorch C++ frontend. Currently, the repo implements only `FaceLandmarkImg` executable. The work  

# **Dependency Installation**
**Follow installation steps in the [wiki](https://github.com/saandeepa93/TorchOpenFace/wiki/Unix-Setup) page. Update the CMakeLists.txt files accordingly.**

## **TorchFace Installation**
* Follow the usual build process to generate `libTorchFace.so` library under `/build/lib/TorchFace/` directory

```
git clone https://github.com/saandeepa93/TorchOpenFace.git
cd TorchFace
mkdir build
cd build
cmake ..
make
```

## **Usage in Python/PyTorch**

### **Loading OpenFace c++ shared object**
```
# Pass OpenFace arguments
openface_args = [model_dir, '-wild', '-mloc', './models/model/main_ceclm_general.txt', '-out_dir', dest_dir]
misc_args = {
  "vis": True, 
  "rec": True,
  "first_only": True
}

# Load c++ shared object
torch.classes.load_library(lib_path)
OpenFace = torch.classes.TorchFaceAnalysis
```

### **FaceLandmarkImg**
```

# Load batch of Image
trans = transforms.ToTensor()
img = Image.open("../data/sample3.png").convert('RGB')
img = trans(img)
img = img.unsqueeze(0).repeat(2, 1, 1, 1) # Sample Batch 
ex_args = {
  "bbox": [[204., 136., 326., 283.], [204., 136., 326., 288.]], 
  "fname": ["saandeep2", "saande3"]
}

# Load FaceLandmarkImg Class
obj = OpenFace.TorchFaceLandmarkImg(openface_args, misc_args)
features = obj.ExtractFeatures(img.clone().detach().cpu().contiguous(), ex_args)
for keys, values in features.items():
  print(f"{keys}: {values.size()}")
```


### **FeatureExtraction Executable**
```
# Load a video
vid_path = os.path.join(root_dir, "data/video01.mp4")
vr = VideoReader(vid_path, ctx=cpu(0))
frame_lst = random.sample(list(range(0, len(vr))), 50)
frame_lst.sort()
frames = vr.get_batch(frame_lst)
frames = frames.asnumpy()
video = torch.from_numpy(frames)
video = video.permute(0, 3, 1, 2)
video = torch.div(video, 255.)
ex_args = {
  "fname": ["video01"],
  'frame_lst': frame_lst
  }
  
# Load Feature Extraction Class
obj = OpenFace.TorchFeatureExtraction(openface_args, misc_args)
features = obj.ExtractFeatures(video.clone().detach().cpu().contiguous(), ex_args)
```

### **Using DataLoader (code snippet)**
```
  obj = OpenFace.TorchFeatureExtraction(openface_args, misc_args)

  test_set = RafDb(cfg, "val", transform, transform)
  test_loader = DataLoader(test_set, batch_size=32, shuffle=False,      num_workers = cfg.DATASET.NUM_WORKERS)
  
  bbox = [0., 0., float(cfg.DATASET.IMG_SIZE), float(cfg.DATASET.IMG_SIZE)]
  for b, (img, _, fnames) in enumerate(tqdm(test_loader)):
    # BBOX (0, 0, max, max) since RAF-DB images are already aligned!
    ex_args = {
      "bbox":[bbox for _ in range(img.size(0))],
      "fname": list(fnames)
    }
    obj.ExtractFeatures(img.cpu().detach(), ex_args)
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


