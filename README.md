# Latest Model
The latest trained model can be downloaded from [here](https://drive.google.com/open?id=1o406WpzPczS1AGLODWQH3-CdfxwxaLKr)

This model is based on GPU training and iterated 100,000 times; Accuracy is about 80%

# Silverpond AI Internship Test

**The Univeristy of Melbourne**

**Ming Jin**

**<mingj2@student.unimelb.edu.au>**

# CIFAR10-TensorFlow-Demo

The chosen topic: *Playing with convolutions*

This simplified version is based on my previous works, as well as the model that used in this repository, please [click here](https://github.com/KimMeen/CIFAR10-Tensorflow-Single-Image-Test) to find out more.

## Introduction
This project aims to build a multi-classes classifier based on CNN and CIFAR10 dataset.

The primary goal in this project is not to build a classifier with an extremely high accuracy, but to predict on a single
image by using the model that we trained, i.e. [A deep learning object detector demo](http://silverpond.com.au/object-detector) in Silverpond.
## Training and Evaluation
The training accuracy and loss curves are shown below.

![Training_Accuracy](https://github.com/KimMeen/CIFAR10-TensorFlow-Demo/blob/master/images/acc.png)

![Training_Loss](https://github.com/KimMeen/CIFAR10-TensorFlow-Demo/blob/master/images/loss.png)

**The prediction results follow this format:**

<br>

*INFO:tensorflow:Restoring parameters from ./model/model.ckpt*

*Model has been restored.*

*-> The input image is : ./test_images/jet2.jpg*

*-> The input label is : plane*

*-> The output(predicted) label is : plane*

*The prediction is correct!*

*The confidence rate is: [ 0.99964237]*

<br>

**The picture used in the demo above(./test_images/jet2.jpg):**

<img src="https://github.com/KimMeen/CIFAR10-TensorFlow-Demo/blob/master/test_images/jet2.jpg" width="450">

## How To Use
There are two steps(simplified) in project: Step1-Training and Step2-Prediction.

You can re-train the model by running Step1-Training.ipynb

You can use the model to predict on a single image directly by using Step2-Prediction.ipynb, and skip the Step1 directly. This step will restore the model that I trained before in ./model folder.

## Future Improvements

Due to the limitation of time, I would like to do these imporvements in the future:

+ Trying to add the objects' bounding boxes in the picture by using [OpenCV](https://opencv.org/) or [labelImg](https://github.com/tzutalin/labelImg)

+ Trying to improve the accuracy by adapting a more complex network, or adjust the parameters

+ The loss curve here is a bit of tricky and I would like to figure out WHY

+ The model was trained on my laptop(CPU based only), and the training process is very excruciating, so I would like to try on GPU(CUDA or others) to speed up.
