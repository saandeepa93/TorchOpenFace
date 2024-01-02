

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