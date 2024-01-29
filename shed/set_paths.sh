#!/bin/bash
echo export CMAKE_PREFIX_PATH="data/scanavan/saandeep/OpenFace2.0/envs/openface" >> ~/.bashrc
echo export CUDA_NVCC_EXECUTABLE="/apps/cuda/cuda-12.1/bin/nvcc" >> ~/.bashrc
echo export CUDA_HOME="/apps/cuda/cuda-12.1" >> ~/.bashrc
echo export CUDNN_INCLUDE_PATH="/apps/cuda/cuda-12.1/include/" >> ~/.bashrc
echo export CUDNN_LIBRARY_PATH="/apps/cuda/cuda-12.1/lib64/" >> ~/.bashrc
echo export LIBRARY_PATH="/apps/cuda/cuda-12.1/lib64" >> ~/.bashrc
source ~/.bashrc