---
classes: wide
title: "Darknet for Ubuntu"
date: 2020-08-03 14:17:00 -0400
categories: deep-learning object-detection darknet
---

# 목차
[0. Darknet Framework](#darknet-framework)   
[1. Install](#install)   
&nbsp;&nbsp;&nbsp;&nbsp;[1.1 Ubuntu](#ubuntu)   
&nbsp;&nbsp;&nbsp;&nbsp;[1.2 NVIDIA](#nvidia)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1.2.1 Driver](#driver)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1.2.2 CUDA](#cuda)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1.2.3 cuDNN](#cudnn)   
&nbsp;&nbsp;&nbsp;&nbsp;[1.3 CMake](#cmake)   
&nbsp;&nbsp;&nbsp;&nbsp;[1.4 GCC](#gcc)   
&nbsp;&nbsp;&nbsp;&nbsp;[1.5 Python](#python)   
&nbsp;&nbsp;&nbsp;&nbsp;[1.6 OpenCV](#opencv)   
&nbsp;&nbsp;&nbsp;&nbsp;[1.7 Darknet](#darknet)   
[2. Test](#test)   
[3. Training on Custom Dataset](#training-on-custom-dataset)   
<br>

# Darknet Framework
More Details : [Pjreddie](https://pjreddie.com), [AlexeyAB](https://github.com/AlexeyAB/darknet)
<br>

# Install

## Ubuntu
- Install [Ubuntu](https://ubuntu.com)
- Update and Upgrade
```shell
$ sudo apt-get update && sudo apt-get upgrade
```

## NVIDIA

### Driver
- Install NVIDIA Driver
```shell
$ sudo ubuntu-drivers autoinstall
```
- Check
```shell
$ nvidia-smi
```

### CUDA
- Install [CUDA10.0](https://developer.nvidia.com/cuda-toolkit-archive)
- Add [PATH](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) in `/.bashrc`
```shell
$ vi /.bashrc
```
```bash
. . .
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
- Check
```shell
$ nvcc --version
```

### cuDnn
- Install [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) >= 7.0 for CUDA10.0
- [Guide for Tar/Debian/RPM](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar)
- Check
```shell
$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

## CMake
- Install CMake >= 3.12
```shell
$ sudo apt-get install cmake
```
- Check
```shell
$ cmake --version
```

## GCC
- Install GCC
```shell
$ sudo apt-get install gcc
```
- Check
```shell
$ gcc --version
$ gcc -v
```

## Python
- Install Python
```shell
$ sudo apt-get install python3 python3-dev python2.7 python2.7-dev
```
- Install Libraries
```shell
$ sudo pip3 install numpy matplotlib
```

## OpenCV
- Remove OpenCV in your computer
```shell
$ sudo apt-get remove libopencv*
$ sudo apt-get autoremove
$ sudo find /usr/local/ -name “*opencv*” -exec rm -rf {} \;
```
- Install Development Tool
```shell
$ sudo apt-get install build-essentail unzip pkg-config
```
- Install Package for Image I/O
```shell
$ sudo apt-get install libjpeg-dev libpong-dev libtiff-dev
```
- Install Package for Video I/O
```shell
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev v4l-utils libxvidcore-dev libx264-dev libxine2-dev
```
- Install Library for Video Streaming
```shell
$ sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```
- Install GTK Library
```shell
$ sudo apt-get install libgtk-3-dev
```
- Install OpenGL Library
```shell
$ sudo apt-get install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev
```
- Install Optimization Library for OpenCV
```shell
$ sudo apt-get install libatlas-base-dev gfortran libeigen3-dev
```
- Download and Extract OpenCV
```shell
$ mkdir opencv
$ cd opencv
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.0.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.0.zip
$ unzip opencv.zip && unzip opencv_contrib.zip
```
- Build OpenCV
```shell
$ cd opencv-3.4.0
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=OFF \
-D WITH_IPP=OFF \
-D WITH_1394=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_DOCS=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D WITH_QT=OFF \
-D WITH_GTK=ON \
-D WITH_OPENGL=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \
-D WITH_V4L=ON  \
-D WITH_FFMPEG=ON \
-D WITH_XINE=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D PYTHON2_INCLUDE_DIR=/usr/include/python2.7 \
-D PYTHON2_NUMPY_INCLUDE_DIRS=/usr/lib/python2.7/dist-packages/numpy/core/include/ \
-D PYTHON2_PACKAGES_PATH=/usr/lib/python2.7/dist-packages \
-D PYTHON2_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so \
-D PYTHON3_INCLUDE_DIR=/usr/include/python3.6m \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include/  \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
../
```
- Compile
```shell
$ sudo make -j4 && sudo make install
$ sudo ldconfig
```
- Check
```shell
$ pkg-config --modversion opencv
```

## Darknet
- Clone
```shell
$ git clone https://github.com/AlexeyAB/darknet
```
- Modify
```shell
$ vi Makefile
```
```
GPU=1 
CUDNN=1
CUDNN_HALF=0 
OPENCV=1 
AVX=0
OPENMP=0 
LIBSO=0
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
# set GPU=1 and CUDNN=1 to speedup on GPU
# set CUDNN_HALF=1 to further speedup 3x times (Mixed-precision on Tensor Cores) GPU: Volta, Xavier, Turing and higher
# set AVX=1 and OPENMP=1 to speedup on CPU (if error occurs then set AVX=0)
# set ZED_CAMERA=1 to enable ZED SDK 3.0 and above
# set ZED_CAMERA_v2_8=1 to enable ZED SDK 2.x
```
- Compile
```shell
$ make -j4
```

# Test

## OpenCV
```shell
$ ./darknet imtest data/eagle.jpg
```

## Darknet
### Image
```shell
$ ./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```
### Video
```shell
$ ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <video-file>
```
### USB Webcam
```shell
$ ./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -c <num>
```

# Training on Custom Dataset