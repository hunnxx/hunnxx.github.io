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
[4. References](#references)
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
- Install [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) >= 7.0 for CUDA10.0 ([Guide for Tar/Debian/RPM](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar))
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
<br>


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
## Pretraind Weights

|**Model**|**Weights**|
|:---:|:---:|
|yolov4.cfg|[yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)|
|yolov4-tiny.cfg|[yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)|
|yolov3.cfg|[yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)|
|yolov3-tiny.cfg|[yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)|

<br>


# Training on Custom Dataset
## Annotation([Yolo-Mark](https://github.com/AlexeyAB/Yolo_mark))
![Annotation](/resources/images/annotation.jpg "Annotation")
```
<object-class> <x_center> <y_center> <width> <height>
```
### Make `train.txt`
```
data/obj/train/img1.jpg
data/obj/train/img2.jpg
data/obj/train/img3.jpg
data/obj/train/img4.jpg
. . .
```
### Make `test.txt`
```
data/obj/test/img1.jpg
data/obj/test/img2.jpg
data/obj/test/img3.jpg
data/obj/test/img4.jpg
. . .
```
## Cfg
### Modify `*.cfg`
```shell
$ cp yolov3.cfg yolov3-custom.cfg
$ vi yolov3-custom.cfg
```
```
[net]
Batch = 64 
Subdivisions = 16
max-batches = classes * 2000
steps = max-batches * 0.8, max-batches * 0.9
width, height = 32의 배수
. . .
[convolutional]
filters = (classes + 5) * 3
. . .
[yolo]
classes = object의 수
. . .
[convolutional]
filters = (classes + 5) * 3
. . .
[yolo]
classes = object의 수
. . .
[convolutional]
filters = (classes + 5) * 3
. . .
[yolo]
classes = object의 수
. . .
```
## Pretrained Weights

|**Model**|**Weights**|
|:---:|:---:|
|yolov4[-custom].cfg|[yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137)|
|yolov4-tiny[-31/-custom].cfg|[yolov4-tiny.conv.29](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29)|
|csresnext50-panet-spp.cfg|[csresnext50-panet-spp.conv.112](https://drive.google.com/file/d/16yMYCLQTY_oDlCIZPfn_sab6KD3zgzGq/view?usp=sharing)|
|yolov3[-spp].cfg|[darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)|
|yolov3-tiny[-prn].cfg|[yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)|
|enet-coco.cfg|[enetb0-coco.conv.132](https://drive.google.com/file/d/1uhh3D6RSn0ekgmsaTcl-ZW53WBaUDo6j/view?usp=sharing)|

## Data/Names
### Make `obj.data`
```
classes=3 # number of classs
train=data/train.txt # path for train.txt file
valid=data/test.txt # path for test.txt file
names=data/obj.names # path for obj.names file
backup=backup/ # path for saving weights
```
### Make `obj.names`
```
Class_name_1
Class_name_2
Class_name_3
. . .
```

## Train
```shell
$ ./darknet detector train <obj.data> <cfg> <pre-trained weights> -map
```
![Log](/resources/images/log.jpg "Log")
### When Should I Stop Training
- 일반적으로 각 클래스의 학습을 위해 2000 iterations 수행 (최소 6000 iterations까지 유지)
-	학습 과정 중 다음과 같이 출력되는 에러 중, 0.XXXXXX avg가 더 이상 감소하지 않을 때, 학습 종료(마지막 평균 loss는 데이터에 따라 0.05에서 3.0으로 구성 가능함)
```
Region Avg IOU: 0.798363, Class: 0.893232, Obj: 0.700808, No Obj: 0.004567, Avg Recall: 1.000000, count: 8 
Region Avg IOU: 0.800677, Class: 0.892181, Obj: 0.701590, No Obj: 0.004574, Avg Recall: 1.000000, count: 8

9002: 0.211667, 0.60730 avg, 0.001000 rate, 3.868000 seconds, 576128 images Loaded: 0.000000 seconds
```
### Select a Weights File for Test
-	학습이 종료되면 obj.data에 지정된 backup 경로에 가중치 파일들이 저장됨
-	Overfitting을 피해 최적의 Weigths 파일을 선택
    - Overfitting : 학습 데이터를 검출 가능하지만 새로운 데이터에 대해 검출 불가능
    - Early Stopping Point : 특정 시점에 학습을 종료함으로써 Overfitting을 방지하는 지점
![Stop](/resources/images/stop.png "Stop")
-	학습 종료 iteratinos 시점부터 이전의 Weights 파일을 다음 명령어를 통해 가장 높은 mAP(또는 IoU)를 가지는 Weights 선택
```
$ ./darknet detector map <obj.data> <cfg> <weights>
```
![mAP](/resources/images/map.jpg "mAP")
<br>


# References
### Darknet
- https://pjreddie.com/darknet/
- https://github.com/AlexeyAB/darknet
- https://github.com/AlexeyAB/Yolo_mark
- https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?-source=friends_link&sk=6039748846bbcf1d960c3061542591d7
- https://www.youtube.com/watch?v=5pYh1rFnNZs
- https://www.youtube.com/watch?v=sUxAVpzZ8hU
### OS
- https://ubuntu.com
- https://www.microsoft.com/ko-kr/software-download
### NVIDIA
- https://www.nvidia.co.kr/Download/index.aspx?lang=kr
- https://developer.nvidia.com/cuda-toolkit
- https://developer.nvidia.com/cuda-toolkit-archive
- https://developer.nvidia.com/cudnn
- https://developer.nvidia.com/rdp/cudnn-archive
- https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
### OpenCV
- https://opencv.org/releases/
- https://github.com/opencv/opencv_contrib
- https://j-remind.tistory.com/57?category=693866
### CMake
- https://cmake.org/
### Python
- https://www.python.org/downloads/windows/
### Visual Studio
- https://visualstudio.microsoft.com/ko/
