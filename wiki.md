
# **Installation**

### **Pytorch/Torchlib Setup**

#### **Some Background**
To ensure you can build a Torchlib library and use it in your correspoding `.py` file later, it is important to ensure compatibility between `PyTorch` and `Torchlib` installation.

Typically, the precompiled binaries of PyTorch (via pip/conda) is sufficient to start using torchlib. But, this installation does not support any external C++ libraries such as OpenCV, OpenBLAS, Boost, which used by OpenFace tool.

To support these external libraries, the torchlib libraries must be `CXX_ABI` enabled. These are available in the [link](https://pytorch.org/get-started/locally/) and needs to be compiled from source. Select the `cxx11 ABI` version and follow the steps mentioned in the [tutorial](https://pytorch.org/cppdocs/installing.html). 

However, This wont support using the shared objects in the PyTorch files (`.py`) because PyTorch by default is `CXX_ABI` disabled. The only way to enable is to build PyTorch from source. 

**Therefore, you should build the latest PyTorch from source and enable `CXX_ABI` manually. Once you build PyTorch from source, you donot need a separate installation of libtorch.**

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
SET(CMAKE_PREFIX_PATH <PyTorch-install_directory>)
find_package(Torch REQUIRED)
```


## **OpenFace Dependency Installation without sudo**
* Sometimes, the machines we work on does not allow us to install any libraries in the install paths (`sudo apt-get <some-lib>`). OpenFace is one such tool. However, you can circumvent this by installing its three key components (dlib, OpenCV, OpenBLAS) locally from source. This assumes that the machine has gcc compiler and other basic dependencies available, which is the case for most Linux machines.


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
-DOpenBLAS_INCLUDE_DIR=/data/scanavan/saandeep/OpenFace/installs/OpenBLAS/include \
-DOpenBLAS_LIB=/data/scanavan/saandeep/OpenFace/installs/OpenBLAS/lib/libopenblas.so \
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
          PATHS "<your-install-dir>/lib/cmake/opencv4/" NO_DEFAULT_PATH )
  ```

## **OpenBLAS Installation**
* You need gfortran compiler to compile OpenBLAS from source. Check whether your machine has it.
```
gfortran --version
```

if the above command returns an error, install it by running.
```
conda install -c conda-forge gfortran_linux-64
```

* you can find the install path of gfortran by 
```
which gfortran
```

* Build OpenBLAS from source.
```
# download the OpenBLAS source
git clone https://github.com/xianyi/OpenBLAS

# compile the library
cd OpenBLAS
make FC=<output-of-which-gfortran>

```
* If the make command throws a `TARGET Error`, it needs a specific cpu TARGET. The most common CPU TARGET is `NEHALEM` for intel. You can try that.
```
make TARGET=NEHALEM FC=<output-of-which-gfortran>
```

* Install the library
```
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
* Due to version change in OpenBLAS installation, the libraries are found in `OpenBLAS_LIBRARIES` instead of `OpenBLAS_LIB`. Update this variable in any CMakeList file while linking OpenBLAS libraries

```
target_link_libraries(LandmarkDetector PUBLIC ${OpenCV_LIBS} ${OpenBLAS_LIBRARIES})
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


# **STEPS in GAIVI**

## **Conda**

* It is important to run all compilations within conda. This isolates any dependency on system libraries which might be of previous versions.

* Create conda environment in a public directory so that it can be accessible by other users without need for separate compilation.
```
conda create -p <public-dir>/openface python=3.10
conda activate <public-dir>/openface
```

* Install cmake, gcc, and gfortran within conda
```
conda install -y cmake

conda install -c anaconda -y gcc_linux-64
conda install -c anaconda -y gxx_linux-64

conda install -c conda-forge -y gfortran
```

* While running any cmake command, use conda installed C++ compilers
```
which x86_64-conda_cos6-linux-gnu-g++ (versions may vary)
cmake -DCMAKE_CXX_COMPILER=/path/to/your/compiler ..
```

* For OpenCV compilation, it is also necessary to install LAPACK (MKL)
```
conda install -y mkl mkl-include
```

* Create public install dirs
```
mkdir <public-dir>/opencv
mkdir <public-dir>/OpenBLAS
mkdir <public-dir>/dlib
```

## **OpenBLAS**

* clone and build OpenBLAS from source. Specify a CPU target explicitly
```
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make TARGET=NEHALEM FC=<output-of-`which gfortran`>
```

* Install in a given path
```
make PREFIX=<OpenBLAS-install-dir> install
```

* Update system paths
```
export LD_LIBRARY_PATH=<OpenBLAS-install-dir>/lib/:$LD_LIBRARY_PATH
```

## **OpenCV**
export LD_LIBRARY_PATH=/home/saandeepaath-admin/anaconda3/envs/cpp/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH



