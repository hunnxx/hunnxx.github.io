---
classes: wide
title: "Tensorflow vs PyTorch"
date: 2020-08-04 10:25:00 -0400
categories: deep-learning framework tensorflow pytorch cnn fashion-mnist
---

# 목차
[0. Framework for Deep-Learning](#framework-for-deep-learning)   
[1. Comparision](#comparision)   
[2. References](#references)
<br>

# Framework for Deep-Learning
### Tensorflow
Tensorflow는 연구 목적으로 Google Brain Team에서 개발한 머신러닝 오픈 소스 플랫폼이다. 심볼릭 수학 라이브러리이면서 인공 신경망 알고리즘과 같은 머신 러닝 응용프로그램에 사용된다. Android와 iOS같은 모바일 환경은 물론 64비트 Linux/MacOS의 데스크탑이나 서버 시스템의 여러 개의 CPU와 GPU에서 구동될 수 있다. 

### PyTorch
PyTorch는 Facebook AI Research(FAIR) Lab.에서 개발한  오픈 소스 머신러닝 프레임워크이다. 사용자 친화적인 프론트 앤드(Front-End)와 분산 학습, 다양한 도구와 라이브러리를 통해 유연한 실험 및 효과적인 상용화를 가능하게 한다. 아직까지는 Tensorflow의 사용자가 많지만, 복잡한 구조와 난이도로 인하여 PyTorch의 사용자가 늘고 있는 추세다.   
<br>


# Comparision
## Tensorflow
### Importing the Library
```python
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
```
### Getting the MNIST Dataset
```python
def showImg(img):   
    plt.imshow(img)

fashion_mnist = keras.datasets.fashion_mnist
(train_imgs_tf, train_labels_tf), (test_imgs_tf, test_labels_tf) = fashion_mnist.load_data()

showImg(train_imgs_tf[0])
print(train_labels_tf[0])
```
### Building the Model
```python
modeltf = keras.Sequential([   
    keras.layers.Conv2D(input_shape=(28,28,1), filters=6, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    keras.layers.Conv2D(16, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu),
    keras.layers.AveragePooling2D(pool_size=2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation=tf.nn.relu),
    keras.layers.Dense(84, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
### Visualizing the Model
```python
modeltf.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
modeltf.summary()
```
### Training
```python
train_imgs_tensorflow = (train_imgs_tf / 255.0).reshape(train_imgs_tf.shape[0], 28, 28, 1)
test_imgs_tensorflow = (test_imgs_tf / 255.0).reshape(test_imgs_tf.shape[0], 28, 28 ,1)
train_labels_tensorflow=keras.utils.to_categorical(train_labels_tf)
test_labels_tensorflow=keras.utils.to_categorical(test_labels_tf)

modeltf.fit(train_imgs_tensorflow, train_labels_tensorflow, epochs=30, batch_size=32)
```
### Result
```python
predictions = modeltf.predict(test_imgs_tensorflow)
correct = 0
for i, pred in enumerate(predictions):
    if np.argmax(pred) == test_labels_tf[i]:
        correct += 1
print('Test Accuracy of the model on the {} test images: {}% with TensorFlow'.format(test_imgs_tf.shape[0], 100 * correct/test_imgs_tf.shape[0]))
```
![result_tensorflow](/resources/images/result_tensorflow.png "result_tensorflow")

## PyTorch
### Importing the Library
```python
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms
```
### Getting the MNIST Dataset
```python
def imshowPytorch(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root='./data/',
                                             train=True, 
                                             transform=transforms,
                                             download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='.data/',
                                             train=False, 
                                             transform=transforms,
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=32, 
                                           shuffle=False)
                                           
data_iter = iter(train_loader)
images, labels = data_iter.next()
imshowPytorch(torchvision.utils.make_grid(images[0]))
print(labels[0])
```
### Building the Model
```python
class NeuralNet(nn.Module):
    def __init__(self, num_of_class):
        super(NeuralNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc_model = nn.Sequential(
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(84, 10)

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(-1, 16*5*5)
        x = self.fc_model(x)
        x = self.classifier(x)
        return x
```
### Visualizing the Model
```python
modelpy = NeuralNet(10)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(modelpy.parameters())   

modelpy
```
### Training
```python
for e in range(30):
    losss = 0.0
    number_of_sub_epoch = 0

    for imgs, labels in train_loader:
        out = modelpy(imgs)
        loss = criterion(out, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losss += loss.item()
        number_of_sub_epoch += 1
    print("Epoch {}: Loss: {}".format(e, losss / number_of_sub_epoch))
```
### Result
```python
correct = 0
total = 0
modelpy.eval()
for imgs, labels in test_loader:
    outputs = modelpy(imgs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the {} test images: {}% with PyTorch'.format(total, 100 * correct // total))
```
![result_pytorch](/resources/images/result_pytorch.png "result_pytorch")
<br>


# References
- https://towardsdatascience.com/tensorflow-vs-pytorch-convolutional-neural-networks-cnn-dd9ca6ddafce
- https://www.tensorflow.org
- https://pytorch.org
