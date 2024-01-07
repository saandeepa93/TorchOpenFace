
# **Installation**

### **OpenFace Dependency Installation**
Follow the detailed steps of [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation#dependency-installation) dependency installation. *Skip the dlib installation step as it needs to be installed from source. Additionally, you do not need to actually install OpenFace, just the dependencies*

### **Pytorch/Torchlib Setup**

#### **Some Background**
To ensure you can build a Torchlib library and use it in your correspoding `.py` file later, it is important to ensure compatibility between `PyTorch` and `Torchlib` installation.

Typically, the precompiled binaries of PyTorch (via pip/conda) is sufficient to start using torchlib. But, this installation does not support any external C++ libraries such as OpenCV, OpenBLAS, Boost, which used by OpenFace tool.

To support these external libraries, the torchlib libraries must be `CXX_ABI` enabled. These are available in the [link](https://pytorch.org/get-started/locally/) and needs to be compiled from source. Select the `cxx11 ABI` version and follow the steps mentioned in the [tutorial](https://pytorch.org/cppdocs/installing.html). 

However, This wont support using the shared objects in the PyTorch files (`.py`) because PyTorch by default is `CXX_ABI` disabled. The only way to enable is to build PyTorch from source. 

**Therefore, you should build the latest PyTorch from source and enable `CXX_ABI` manually. Once you build PyTorch from source, you donot need a separate installation of libtorch as well.**

### **Build PyTorch from source**
* Ensure you have a clean installation of CUDA and nvcc. Check both versions by running the below command. Any error in either commands means they are not installed correctly.
```
nvidia-smi
nvcc --version
```
 

* Follow the steps mentioned in [link](https://github.com/pytorch/pytorch#from-source) build PyTorch from source. You might encounter some warnings which should be ok.

* Ensure running all commands inside an anaconda environment.

* Once installed, to access torchlib inside your cpp file, simply edit the `CMAKE_PREFIX_PATH` to directory where PyTorch was cloned to and built
```
SET(CMAKE_PREFIX_PATH <PyTorch-install_directory)
find_package(Torch REQUIRED)
```

## **Dlib Installation (credits to [perplexity](https://www.perplexity.ai/))**
  #### If you try to run dlib libraries, it throws an error. For this reason, you have to compile dlib from source and additionally install it locally (to remove sudo dependency). To compile dlib as a shared library using CMake  without sudo, you can follow these steps:

  * Download and extract dlib: You can download the latest version of dlib from the official [website](https://github.com/davisking/dlib)
  
  * Before running cmake, add the below line to `CMakeLists.txt` of dlib build file on top.
  ```
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  ```

  * After downloading, extract the files to a directory.
  Create a build directory: Navigate to the dlibdirectory and create a new directory named build:

  ```
  cd dlib
  mkdir build
  cd build
  ```

  * Run CMake with the -DBUILD_SHARED_LIBS=1 option: This option tells CMake to build dlib as a shared library. Run the following command:

  ```
  cmake -DBUILD_SHARED_LIBS=1 ..
  ```

  * Compile dlib: After running CMake, compile dlib using the make command:
  ```
  make
  ```

  * Install dlib locally: Instead of using `sudo make install` which installs dlib system-wide, you can install dlib locally in a directory of your choice. For example, if you want to install dlib in a directory named local_install in your home directory, you can do so with the following commands: 
  This will install dlib in ~/local_install/usr/local/lib and ~/local_install/usr/local/include2

  ```
  make install DESTDIR=~/local_install
  ```

  * To use the shared library in another project, you need to tell the linker where to find it. You can do this by adding the path to the library to the `LD_LIBRARY_PATH` environment variable:

  ```
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/local_install/usr/local/lib
  ```

  * Next, edit the `CMakeLists.txt` in the main project as follows
  ```
  find_package(dlib 19.13 REQUIRED PATHS "<path-installed>/usr/local/lib/cmake/dlib" NO_DEFAULT_PATH)
  ```

## **OpenCV Installation**
  * Go to the OpenCV releases page and download the source code for OpenCV 4.1.
  ```
  wget https://github.com/opencv/opencv/archive/4.1.0.zip
  unzip 4.1.0.zip
  cd opencv-4.1.0
  mkdir build
  cd build
  ```

  * Run CMAKE 
  ```
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=<your-install-dir> \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D WITH_CUDA=OFF \
  ..
  ```

  * Compile and install
  ```
  make -j8 && make install
  ```

  * Configure Your Environment:
  ```
  export LD_LIBRARY_PATH=<your-install-dir>/lib:$LD_LIBRARY_PATH
  export PKG_CONFIG_PATH=<your-install-dir>/lib/pkgconfig:$PKG_CONFIG_PATH
  ```

  * In your CMakeLists.txt, find OpenCV using the following line
  ```
  find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect
          PATHS "<<your-install-dir>>/lib/cmake/opencv4/" NO_DEFAULT_PATH )
  ```


## **OpenBLAS Installation**

* Build OpenBLAS from source.
```
# download the OpenBLAS source
git clone https://github.com/xianyi/OpenBLAS

# compile the library
cd OpenBLAS && make FC=gfortran

# install the library
make PREFIX=<your-install-dir> install
```

* Update system path information
```
export LD_LIBRARY_PATH=<your-install-dir>/lib/:$LD_LIBRARY_PATH
```

  * In your CMakeLists.txt, find OpenCV using the following line
  ```
  find_package(OpenBLAS REQUIRED PATHS "<your-install-dir>/lib/cmake/openblas" NO_DEFAULT_PATH)
  ```


## **TorchFace Installation**
* Follow the usual build process to generate `libTorchFace.so` library under `/build/lib/TorchFace/` directory
```
cd TorchFace
mkdir build
cd build
cmake ..
make
```

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX='/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external' \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D WITH_CUDA=OFF \
..

export LD_LIBRARY_PATH=/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external/lib/pkgconfig:$PKG_CONFIG_PATH

 cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external -D BUILD_TIFF=ON -D WITH_TBB=ON -D BUILD_SHARED_LIBS=ON \
-D OPENCV_EXTRA_MODULES_PATH=/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external/opencv_shared/ ..


export LD_LIBRARY_PATH=/home/saandeepaath-admin/projects/learning/cpp_cmake/example3/external/OpenBLAS/lib/:$LD_LIBRARY_PATH

