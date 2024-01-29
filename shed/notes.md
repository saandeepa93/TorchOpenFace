ROOT INSTALL DIR
```
/data/scanavan/saandeep/OpenFace2.0/installs/
```

# **CONDA**
```
conda create -p /data/scanavan/saandeep/OpenFace2.0/envs/openface python=3.10
conda activate /data/scanavan/saandeep/OpenFace2.0/envs/openface
```

* Install GCC depending on the PyTorch + CUDA version you would like to use. For e.g., If you have an older CUDA version and building the latest PyTorch from source, then GNU compiler will throw an error.
```
conda install gcc_linux-64=11.2.0 gxx_linux-64=11.2.0
conda install cmake tbb jpeg libpng libtiff mkl mkl-include
EXPORT CC=/data/scanavan/saandeep/OpenFace2.0/envs/openface/bin/x86_64-conda-linux-gnu-cc
EXPORT CXX=/data/scanavan/saandeep/OpenFace2.0/envs/openface/bin/x86_64-conda-linux-gnu-c++
source ~/.bashrc
```


# **OpenBLAS**
```
conda install -c anaconda openblas
```

## **OpenBLAS variables**

* OpenBLAS_LIB: `/data/scanavan/saandeep/OpenFace2.0/envs/openface/lib/libopenblas.so`
* OpenBLAS_INCLUDE_DIR: `/data/scanavan/saandeep/OpenFace2.0/envs/openface/include`

```
export LD_LIBRARY_PATH=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/lib64:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/lib64/pkgconfig:$PKG_CONFIG_PATH
source ~/.bashrc
```


# **OpenCV**

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-"4.1.0" \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D OPENCV_PYTHON3_INSTALL_PATH=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0-py3/lib/python3.5/site-packages \
-D BUILD_TIFF=ON \
-D ENABLE_PRECOMPILED_HEADERS=OFF \
-D BUILD_EXAMPLES=ON ..
```

# **dlib**

```
cmake -DBUILD_SHARED_LIBS=1 ..
make
make install DESTDIR=/data/scanavan/saandeep/OpenFace2.0/installs/dlib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:=/data/scanavan/saandeep/OpenFace2.0/installs/dlib/usr/local/lib64
```

# **OpenFace2.0**
```
cmake \
-D OpenBLAS_LIB=/data/scanavan/saandeep/OpenFace2.0/envs/openface/lib/libopenblas.so \
-D OpenBLAS_INCLUDE_DIR=/data/scanavan/saandeep/OpenFace2.0/envs/openface/include \
-D OpenCV_DIR=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/lib64/cmake/opencv4 \
-D OpenCV_LIBRARIES=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/lib64 \
-D OpenCV_INCLUDE_DIRS=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/include \
-D dlib_DIR=/data/scanavan/saandeep/OpenFace2.0/installs/dlib/usr/local/lib64/cmake/dlib \
..
```

# **Build PyTorch+CUDA from source**
1. Versions!

```
CUDA: 12.1
GCC: 11.2 (>=9.4)
PyTorch 2.3
```

```
echo export CMAKE_PREFIX_PATH="data/scanavan/saandeep/OpenFace2.0/envs/openface" >> ~/.bashrc
echo export CUDA_NVCC_EXECUTABLE="/apps/cuda/cuda-12.1/bin/nvcc" >> ~/.bashrc
echo export CUDA_HOME="/apps/cuda/cuda-12.1" >> ~/.bashrc
echo export CUDNN_INCLUDE_PATH="/apps/cuda/cuda-12.1/include/" >> ~/.bashrc
echo export CUDNN_LIBRARY_PATH="/apps/cuda/cuda-12.1/lib64/" >> ~/.bashrc
echo export LIBRARY_PATH="/apps/cuda/cuda-12.1/lib64" >> ~/.bashrc
```

* Export CUDA paths
```
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

* Workound- disable Conda's linkers. Creates errors
```
cd /data/scanavan/saandeep/OpenFace2.0/envs/openface/compiler_compat
mv ld ld-old
```

* Install
```
python setup.py install
```

* Enable conda linkers post installations
```
cd /data/scanavan/saandeep/OpenFace2.0/envs/openface/compiler_compat
mv ld-old ld
```

# **TorchOpenFace**

* Build 
```
cmake \
-D OpenBLAS_LIB=/data/scanavan/saandeep/OpenFace2.0/envs/openface/lib/libopenblas.so \
-D OpenBLAS_INCLUDE_DIR=/data/scanavan/saandeep/OpenFace2.0/envs/openface/include \
-D OpenCV_DIR=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/lib64/cmake/opencv4 \
-D OpenCV_LIBRARIES=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/lib64 \
-D OpenCV_INCLUDE_DIRS=/data/scanavan/saandeep/OpenFace2.0/installs/OpenCV-4.1.0/include \
-D dlib_DIR=/data/scanavan/saandeep/OpenFace2.0/installs/dlib/usr/local/lib64/cmake/dlib \
-D 
..
```


## **Instructions to run OpenFace**

* Activate Conda environment
```
conda activate /data/scanavan/saandeep/OpenFace2.0/envs/openface
```

* Enter build directory of OpenFace
```
cd /data/scanavan/saandeep/OpenFace2.0/OpenFace/build/
```

* Run sample code
```
./bin/FaceLandmarkImg -f ../imgs/sample.png -wild
```

* Check output in `./processed` directory.



## **Instructions to run TorchOpenFace**

* Activate Conda environment
```
conda activate /data/scanavan/saandeep/OpenFace2.0/envs/openface
```

* Enter build directory of OpenFace
```
cd /data/scanavan/saandeep/OpenFace2.0/TorchOpenFace
```

* Run sample code. This should take about 30 seconds to run.
```
sbatch t_run.sh
```

* Check the `slurm-<job-id>.slurm` file to see if there are any error. 


* If no errors, Output should be saved in `./data/processed` directory after some time.


