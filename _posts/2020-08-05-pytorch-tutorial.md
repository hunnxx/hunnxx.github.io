---
classes: wide
title: "Pytorch Tutorial"
date: 2020-08-05 10:00:00 -0400
categories: deep-learning framework pytorch cnn mnist
---

# 목차
[0. Learning Pytorch](#learning-pytorch)   
[1. Example with MNIST](#example-with-mnist)   
[2. References](#references)   
<br>

# Learning PyTorch
## Tensors
Python의 Numpy는 최고의 라이브러리이지만 GPU의 연산을 가속화시키기 위해서 사용할 수는 없다. 최근 DNN (Deep Neural Network) 을 위한 GPU는 50배 또는 그 이상의 속도를 제공하지만 Numpy는 이를 활용하지 못하기 때문에 Deep Learning을 위해서는 적합하지 않다.

Facebook AI Research (FAIR) 팀에서 머신러닝을 위한 개발한 PyTorch의 `Tensor`는 Numpy를 대체할 수 있다. PyTorch의 `Tensor`는 개념적으로 Numpy의 배열 (N-Dimensional) 과 같지만 Deep Learning을 위한 많은 기능을 제공한다.

### Example with Random Data
```python
# ----------------------------------------------------------------------------------------------------
# N: batch size
# D_IN: input dimension
# H: hidden dimension
# D_OUT: output dimension
# lr: learning rate
# x: random input data
# y: random output data
# w1: random weights 1
# w2: random weights 2
# -----------------------------------------------------------------------------------------------------
# torch.mm(...): https://pytorch.org/docs/master/generated/torch.mm.html
# torch.clamp(...): https://pytorch.org/docs/stable/generated/torch.clamp.html
# -----------------------------------------------------------------------------------------------------

import torch   

device = torch.device('cpu') # torch.device('cuda:0')

N, D_IN, H, D_OUT = 64, 1000, 100, 10
lr = 1e-6

x = torch.randn(N, D_IN, device=device, dtype=torch.float)
y = torch.randn(N, D_OUT, device=device, dtype=torch.float)

w1 = torch.randn(D_IN, H, device=device, dtype=torch.float)
w2 = torch.randn(H, D_OUT, device=device, dtype=torch.float)

for t in range(500):   
    # Forward pass
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)
    
    # Backpropagation to compute gradients
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    # Update wegiths
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
```
## Autograd
위의 `Example with Random Data`는 인공 신경망의 Forward/Backward 과정을 모두 직접 구현하였다. 이러한 과정을 모두 구현하는 것이 어려운 일은 아니지만 인공 신경망의 크기가 커짐에 따라 빠르게 어려워질 것이다.

따라서, PyTorch에서는 이러한 Backward 과정의 Gradient를 자동으로 계산해주는 기능을 제공한다. `Autograd` 패키지는 인공 신경망의 Forward 과정에서 모든 Tensor의 연산 과정의 그래프를 생성하고 Loss의 Gradient를 계산한다.
```python
# ----------------------------------------------------------------------------------------------------
# torch.no_grad(): 가중치 갱신 과정에서는 Autograd가 Tensor의 추적을 막기 위함
# tensor.grad.zero(): 가중치 갱신 이후, 각 Tensor의 Gradient 직접 초기화
# ----------------------------------------------------------------------------------------------------
import torch

device = torch.device('cpu') # torch.device('cuda:0')

N, D_IN, H, D_OUT = 64, 1000, 100, 10
lr = 1e-6

x = torch.randn(N, D_IN, device=device, dtype=torch.float)
y = torch.randn(N, D_OUT, device=device, dtype=torch.float)

w1 = torch.randn(D_IN, H, device=device, dtype=torch.float, requires_grad=True)
w2 = torch.randn(H, D_OUT, device=device, dtype=torch.float, requires_grad=True)

for t in range(500):   
    # Forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Backpropagation to compute gradients
    loss.backward()

    # Update wegiths
    with torch.no_grad():
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2
        w1.grad.zero_()
        w2.grad.zero_()
```
### Defining new autograd function
```python
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input
        
. . .

for t in range(500):
    relu = MyReLU.apply
    
    y_pred = relu(x.mm(w1)).mm(w2)
    
. . .
```
## nn.Module
`Tensor`와 `Autograd`는 복잡한 연산을 정의하고 자동으로 계산해줌으로써 편의성을 제공하지만 큰 인공 신경망에서는 너무 Low-Level의 기능이다.

인공 신경망의 레이어의 연산을 다룰 때, 레이어의 일부는 학습 동안 최적화되어지는 Learnable Parameters를 가지고 있다. Tensorflow에서는 [Keras](https://github.com/fchollet/keras), [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) 그리고 [TFLearn](http://tflearn.org/)과 같은 패키지가 레이어에 대한 High-Level의 기능을 제공한다. Pytorch에서는 `nn` 패키지가 동일한 기능을 제공한다. `nn` 패키지는 신경망과 대략적으로 같은 Models를 정의하고 있다. 하나의 Module은 Input Tensor를 받아 Output Tensor를 계산하면서 Learnable Parameters를 포함하고 있는 Tensors 같은 내부 상태를 가지고 있다. `nn` 패키지는 또한 학습 과정 중 사용 가능한 Loss Function을 가지고 있다.
```python
# ----------------------------------------------------------------------------------------------------
# torch.nn.MSELoss(...): https://pytorch.org/docs/master/generated/torch.nn.MSELoss.html?highlight=torch.nn.mseloss#torch.nn.MSELoss
# torch.nn.Sequential(...): https://pytorch.org/docs/master/generated/torch.nn.Sequential.html?highlight=torch.nn.sequential#torch.nn.Sequential
# torch.nn.Linear(...): https://pytorch.org/docs/master/generated/torch.nn.Linear.html
# torch.nn.ReLU(...): https://pytorch.org/docs/master/generated/torch.nn.ReLU.html?highlight=torch.nn.relu#torch.nn.ReLU
# ----------------------------------------------------------------------------------------------------
import torch

N, D_IN, H, D_OUT = 64, 1000, 100, 10
lr = 1e-4
loss_fn = torch.nn.MSELoss(reduction='sum') # Mean Squared Error Loss

x = torch.randn(N, D_IN)
y = torch.randn(N, D_OUT)

# nn.Sequential은 다른 모듈을 포함하는 모듈이면서 순차적으로 모듈의 Output을 전달하면서 수행
model = torch.nn.Sequential(
    torch.nn.Linear(D_IN, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_OUT),
)

for t in range(500):
    # Forward pass
    y_pred = model(x)
    
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
```
## optim
이전까지의 과정에서는 Weights에 대한 갱신을 직접 구현하거나 `Autograd` 패키지를 통해 Learnable Parameters를 가짐으로써 수행하였다. 이러한 과정은 Gradient Descent 와 같은 간단한 최적화 과정에서는 쉽게 수행가능하다. 하지만 실제로는 AdaGrad, RMSProp, Adam 등 보다 세련된 최적화 알고리즘을 사용한다. `optim` 이러한 최적화 알고리즘을 제공하는 패키지다.
```python
# ----------------------------------------------------------------------------------------------------
# torch.optim.Adam(...): https://pytorch.org/docs/master/optim.html?highlight=torch.optim.adam#torch.optim.Adam
# ----------------------------------------------------------------------------------------------------
import torch

. . .

# Adam 최적화 알고리즘에게 model.parameters() Tensor를 최적화해야한다고 정의  
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for t in range(500):
    # Forward pass
    y_pred = model(x)
    
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    
    # Backward pass 전, 버퍼에 누적되는 Gradient를 초기화
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    
    # Update wegiths
    optimizer.step()
```
## Custom nn.Modules
```python
# ----------------------------------------------------------------------------------------------------
# torch.optim.SGD(...): https://pytorch.org/docs/master/optim.html?highlight=torch.optim.sgd#torch.optim.SGD
# ----------------------------------------------------------------------------------------------------
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_IN, H, D_OUT):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_IN, H)
        self.linear2 = torch.nn.Linear(H, D_OUT)
        
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
        
. . .

model = TwoLayerNet(D_IN, H, D_OUT)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

. . .

``` 
## Control Flow + Weight Sharing
가중치를 공유하여 계산하는 N개의 Hidden Layers를 구현하는 예제다. 일반적인 Python 반복문을 통해 간단하게 구현한다.
```python
import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_IN, H, D_OUT):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_IN, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_OUT)
        
    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

. . .

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

. . .
```
<br>

# Example with MNIST
## MNIST Data Setup
이번 파트에서는 0 에서 9 사이의 흑백 자필 숫자로 구성된 MNIST 데이터셋을 이용할 것이다. 
### Download MNIST from Web
```python
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
```
### Extract and Show Data
```python
import pickle
import gzip
from matplotlib import pyplot
import numpy as np

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# 1x784 -> 28x28
pyplot.imshow(x_train[0].reshape((28,  28)), cmap="gray")
print(x_train.shape)
```
![test_img_mnist](/resources/images/test_img_mnist.png)
### Convert Numpy to Tensor
```python
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape

print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
```
## Neural Net (No torch.nn)
```python
# ----------------------------------------------------------------------------------------------------
# weights: random weights
# bias: zero bias
# bs: batch size
# loss_func: loss function
# lr: learning rate
# epochs: total epochs
# ----------------------------------------------------------------------------------------------------
# @: dot product operation
# def log_softmax(...): activation function
# def model(...): nueral net
# def nll(...): loss function
# def accuracy(...): accuracy function
# ----------------------------------------------------------------------------------------------------
import math

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
    
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
    
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad()
bias = torch.zeros(10, requires_grad=True)
bs = 64
loss_func = nll
lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n-1) // bs + 1):
        # Forward pass
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        
        # Backward pass
        loss.backward()
        
        # Update Weights
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
        
        print(loss, accuracy(model(xb), yb))
```
## Using torch.nn.Modules
이전의 `Neural Net`에서는 간단한 신경망을 PyTorch의 핵심 기능을 제외하고 구성하였다. 이번에는 nn.Modules의 기능을 통해 이전 코드를 짧고 융하게 바꿀 예정이다. 
### Using torch.nn.functional/Moudule/Linear
```python
import math
import torch.nn.functional as F
from torch import nn

class MNIST_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)
    
    def forward(self, xb):
        return self.lin(xb)

. . . 

# def log_softmax(x):
#    return x - x.exp().sum(-1).log().unsqueeze(-1)

# def model(xb):
#    return xb @ weights * bias
    
# def nll(input, target):
#    return -input[range(target.shape[0]), target].mean()

. . .

# weights = torch.randn(784, 10) / math.sqrt(784)
# weights.requires_grad()
# bias = torch.zeros(10, requires_grad=True)
loss_func = F.cross_entropy
model = MNIST_Logistic()

. . .

def main():
    for epoch in range(epochs):
        for i in range((n-1) // bs + 1):
            
            . . .
            
            # Backward pass
            loss.backward()
            
            # Update weights
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()
                
if __name__ == '__main__':
    main()
```
## Refactor using optim
```python
import math
import torch.nn.functional as F
from torch import nn
from torch import optim

. . .

def get_model():
    model = MNIST_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)
    
. . .

# model = MNIST_Logistic()
model, opt = get_model()

. . .

def main():
for epoch in range(epochs):
    for i in range((n-1) // bs + 1):
        
        . . .
        
        # Backward pass
        loss.backward()
        
        # Update weights
        opt.step()
        opt.zero_grad()
            
. . .
```
## Refactor using Dataset/DataLoader
```python
# ----------------------------------------------------------------------------------------------------
# TensorDataset: https://pytorch.org/docs/master/data.html?highlight=tensordataset#torch.utils.data.TensorDataset
# DataLoader: manage batches in Dataset
# ----------------------------------------------------------------------------------------------------
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

. . .

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

. . .

def main():
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            
            . . .
        
. . .
```
## Add Validation
```python
# ----------------------------------------------------------------------------------------------------
# model.train/eval(): nn.BatchNorm2d/nn.Dropout과 같은 Layers가 있는 모델에게 현재 Phases를 지정
# ----------------------------------------------------------------------------------------------------
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

. . .

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

. . .

def main():
for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
    print(epoch, valid_loss / len(valid_dl)
    
. . .
```
## Create fit() and get_data()
```python
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

. . .

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)
    
def get_data(trina_ds, valid_ds, bs):
    return (
    DataLoader(train_ds, batch_size=bs, shuffle=True), 
    DataLoader(valid_ds, batch_size=bs * 2)
    )

. . .

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

. . .

def main(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

. . .
```
## Switch to CNN
```python
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

. . . 

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))
        
. . .

def get_model():
    model = MNIST_CNN()
    return model, optim.SGD(model.parameters(), lr=lr, momentum=0.9)

. . .

lr = 0.1

. . .
    
```
## nn.Sequential

## Wrapping DataLoader

## Using your GPU

## Conclusion
<br>

# References
- https://pytorch.org
- https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
- https://pytorch.org/tutorials/beginner/nn_tutorial.html
