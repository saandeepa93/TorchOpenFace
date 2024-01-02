# **TorchFace**

### This repository is to integrate [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) with PyTorch C++ frontend. Follow the steps below to setup the repo.

# **Installation**

### **OpenFace Installation**
Follow the detailed steps of [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation#dependency-installation) installation. *Skip the dlib installation step as it needs to be installed from source.*

### **Pytorch/Torchlib Setup**

#### **Some Background**
To ensure you can build a Torchlib library and use it in your correspoding `.py` file later, it is important to ensure compatibility between `PyTorch` and `Torchlib` installation.

Typically, the precompiled binaries of PyTorch (via pip/conda) is sufficient to start using torchlib. But, this installation does not support any external C++ libraries such as OpenCV, OpenBLAS, Boost, which used by OpenFace tool.

To support these external libraries, the torchlib libraries must be `CXX_ABI` enabled. These are available in the [link](https://pytorch.org/get-started/locally/) and needs to be compiled from source. Select the `cxx11 ABI` version and follow the steps mentioned in the [tutorial](https://pytorch.org/cppdocs/installing.html). 

However, This wont support using the shared objects in the PyTorch files (`.py`) because PyTorch by default is `CXX_ABI` disabled. The only way to enable is to build PyTorch from source. 

**Therefore, you should build the latest PyTorch from source and enable `CXX_ABI`. Once you build PyTorch from source, you donot need a separate installation of libtorch as well.**

### **Build PyTorch from source**
* Ensure you have a clean installation of CUDA and nvcc. Check both versions by running the below command. Any error in either commands means they are not installed correctly.
```
nvidia-smi
nvcc --version
```
 

* Follow the steps mentioned in [link](https://github.com/pytorch/pytorch#from-source) build PyTorch from source. You might encounter some warnings which should be ok.

* Ensure running all commands inside an anaconda environment.

* Once installed, to access torchlib inside your cpp file, simply set the `CMAKE_PREFIX_PATH` to directory where PyTorch was cloned to and built
```
SET(CMAKE_PREFIX_PATH <PyTorch-install_directory)
find_package(Torch REQUIRED)
```




# **Build C++ Extension (General)**

* Build PyTorch from source using [link](https://github.com/pytorch/pytorch#from-source)
* Ensure it is `CXX_ABI` enabled. This allows torchlib to work with external C++ libraries like OpenCV, OpenBLAS etc.
* Add the installed PyTorch directory as `CMAKE_PREFIX_PATH` in your `CMakeLists.txt` file

```
SET(CMAKE_PREFIX_PATH /home/saandeepaath-admin/projects/learning/cpp_cmake/example2/pytorch)
find_package(Torch REQUIRED)
```

* Follow the usual build process to generate `.so` library
```
mkdir build
cmake ..
make
```


# **Build libtorch/python/c++ OpenFace**

## **Dlib installation**

  * If you try to run dlib libraries with torchlib included, it throws an error. For this reason, you have to compile dlib from source and additionally install locally (more related to BRIAR requriement)

  * Follow this link to install dlib locally: [perplexity](https://www.perplexity.ai/search/How-to-install-ejY1cIoEQfO_9YGIP0D8Wg?s=c#f290fff8-d03a-45b1-b4d8-649df13719b1)

  * Before running cmake, add the below line to CMakeLists of dlib build file on top.
  ```
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  ```

  * In the project where you want to use dlib, use `find_package` as follows
  ```
  find_package(dlib 19.13 REQUIRED PATHS "<path-installed>/usr/local/lib/cmake/dlib" NO_DEFAULT_PATH)
  ```


## **Updates to OpenFace**

  * Updated model path
  * Added default constructor to `FaceAnalyser` class
  * Added default constructor to `Visualizer` class.
  * Make `SetCameraIntrinsics(float fx, float fy, float cx, float cy)` under `ImageCapture.h` as `public`
  * Added model directory variable to `LandmartDetector` and `FaceAnalyser` classes