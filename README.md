
# 1. Introduction

In this Repo, I will cover fully convolutional networks, object detection, semantics segmentation and inference optimizations. Finally, I'll apply these concepts and techniques to train and supervise semantic segmentation network and test my result on automotive videos. 

A typical convolutional neural network might consist of a series of convolution layers followed by fully connected layers and ultimately a soft max activation function. This is a great architecture for a classification task like is this a picture of a hotdog? 

<p align="right">
<img src="./img/1.png" alt="convolution layers followed by fully connected layers" />
<p align="right">
 
If I want to change my task to answer the question, "where in the picture is the hotdog?". The question is much more difficult to answer since fully connected layers don't preserve spatial information. If I change the classification network  from the connected layers to convolutional layers to create fully convolutional networks or FCN's for short. 

<p align="right">
<img src="./img/2.png" alt="fully convolutional layers" />
<p align="right">
 
FCN's help us answer where is the hotdog question because while doing the convolution they preserve the spatial information throughout the entire network. 

In a classic convolutional network with fully connected final layers, the size of the input is constrained by the size of the fully connected layers. Passing different size images through the same sequence of  convolutional layers and flattening the final output. These outputs would be of different sizes which doesn't bode very well for matrix multiplication. Additionally, since convolutional operations fundamentally don't care about the size of the input, a fully convolutional network will work on images of any size. I will dive more into details of fully convolutional networks next. 


# 2. Fully Convolutional Networks
 Fully Convolutional Networks have achieved state of the art results in computer vision tasks,such as semantic segmentation. FCNs take advantage of three special techniques:

 1.	replace fully connected layers with one by one convolutional layers
 2.	up-sampling through the use of transposed convolutional layers
 3. skip connections.
 
The skip connections allow the network to use information from multiple resolution scales. As a result the network is able to make more precise segmentation decisions. I will discuss these techniques in greater detail shortly. 

Structurally an FCN is usually comprised of two parts (see image below):

<p align="right">
<img src="./img/3.png" width="600" height="300" alt="fully convolutional layers" />
<p align="right">
 
  1.	encoder 
  2.	decoder. 

The encoder is a series of convolutional layers like VGG and ResNet. The goal of the encoder is to extract features from the image. The decoder up-scales the output of the encoder such that it's the same size as the original image. Thus, it results in segmentation or prediction of each individual pixel in the original image. 

 
### 2.1. Fully Connected to 1x1 Convolution

the first special technique in FCNs replaces a fully connected layer(from the classification networks) with one by one convolutional layers. This will result in the output value with the tensor, which has 4D instead, 2D, so spatial information will be preserved. 

Recall, the output of the convolution operation is the result of sweeping the kernel over 
the input with the sliding window and performing element wise multiplication and summation. 

<p align="right">
<img src="./img/4.png" width="600" height="300" alt="Fully Connected to 1x1 Convolution" />
<p align="right">
 
 One way to think about one by one convolutional layers is the number of kernels is equivalent to the number of outputs in a fully connected layer. Similarly, the number of weights in each kernel is equivalent to the number of inputs in the fully connected layer. Effectively, this turns convolutions into a matrix multiplication with spatial information.
 
#### 1x1 Convolution Exercise

In this [exercise](https://github.com/A2Amir/Fully-Convolutional-Networks--FCNs--/blob/master/Code/FullyConnectedto1x1ConvolutionExercise.ipynb) I am going to rewrite a dense layer, tf.layers.dense as a convolutional layer, tf.layers.conv2d. The underlying math will be the same, but the spatial information will be preserved allowing seamless use of future convolutional layers.



