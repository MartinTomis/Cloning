
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (udacity code)
* model1.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video1.mp4 and video2.mp4

####2. Submission includes functional code
Using the Udacity provided simulator and the original drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model1.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used a slight modification of architecture described [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and in the lectures. My model consists of 3 convolutional networks with a 5x5 filter and 2x2 stride, followed by a dropout layer, followed by two layers with a 3x3 filter. Each of the convolutional layers uses a RELU layer to introduce non-linearity. These are followed by fully connected layers. A picture of my architecture, including the dimensions at each layer, is shown below.

####2. Attempts to reduce overfitting in the model

I included one dropout layer between the convolutional layer with 5x5 filter and the one with 3x3 filter. 

To assess potential overfitting, I relied on analyzing the MSE from the training sets and the validation sets. Without the dropout layer, the validation MSE error was usually higher than the learning error. In some models I tried, the validation MSE was usually 50% to 100% above the training error. 

With the dropout rate, the validation error in most epochs actually remained below the training set MSE (X axis is shifted by 1, it should start at 1).

![alt text](https://github.com/MartinTomis/Cloning/blob/master/model1_MSE.png "MSE")

####3. Model parameter tuning

The model used an Adam optimizer, so the learning rate is adjusted automatically.

####4. Appropriate training data

The training data include 3 laps of driving in the middle of the road, approximately 1 lap of recovery driving (recording starts when the car starts returning from the side of the road to the middle) and 1 lap of smooth driving through the curves. Only the track from the lake is used to generate data.

###Architecture and Training Documentation

####1. Solution Design Approach

In the model development, I took the following key steps. 
1. First model was a simple linear regression.
2. Then it was extended to LeNet and flipped images were used. Later, images from all three cameras were used. Different croppings of input images were attempted.
3. NVIDIA model described above was used.
4. Different corrections to the steering angles were attempted (0.2, 0.1, 0.3, finally 0.25 was used).

Most steps were accompanied by generating additional data. I would estimate that each of steps 2-3 above was run on 5 different datasets.

On the driving track 1, I encountered issues on the following segments:
1. left-turn before the bridge.
2. Surprisingly, the bridge - sometimes the car hit the side.
3. Sharp left turn after the bridge.
4. The sharp right turn after segment 3.

Often when I encountered such issue, I generated additional data by recovery driving in the particular segment. But quite often, this led to the car failing in another segment.

In the end, after a slight correction of the angle for the side cameras, the car performed well on a dataset where it previously failed.


####2. Final Model Architecture

My model consists of 3 convolutional networks with a 5x5 filter and 2x2 stride, followed by two layers with a 3x3 filter. Each of the convolutional layers uses a RELU layer to introduce non-linearity. These are followed by fully connected layers.
Here is a visualization of the architecture: dimensions of the layers are shown on the right.

![alt text](https://github.com/MartinTomis/Cloning/blob/master/architecture.PNG "Architecture") 

####3. Creation of the Training Set & Training Process

I generate my own data. The dataset to train this model includes 3 laps of driving in the middle of the road, approximately 1 lap of recovery driving (recording starts when the car starts returning from the side of the road to the middle) and 1 lap of smooth driving through the curves. Only the track from the lake is used to generate data.

Data from all 3 cameras are used, and the images are flipped, to get more images.

In the process of training, I altogether generated over 3 GB of driving data, as I could not find a model that worked. My solution often was to generate additional data from the problematic road segments, if the MSE at the training and validation set were close. Adding recovery driving from the problematic segments often lead to a failure in another segment.

The final dataset has approximately 90 000 images - these are then flipped, essentially doubling the size.

I finally randomly shuffled the data set and put 20% of the data into a validation set. I think that this is a common approach, Tibshirani, Hastie and Friedman write that in a similar context of K-fold cross validation, the dataset is usually split into 5 or 10 equal parts. I used this training data for training the model. The validation set helped determine if the model was over or under fitting / the MSE graph is shown above.

As for the number of epochs, I tried multiple choices. First idea was to use 5. Tibshirani, Hastie and Friedman show a exampe of 5 networks from Yan LeCun's paper, suggesting that the success rate remains esentially flat after approximately 5 epochs. However, mine MSE continued decreasing at a reasonably quick rate after 5 epochs, so it may be desirable to use more. 

Other argument for 5 epochs is that if I did "proper" 5-fold cross validation and used 20% of data for validation, I would do 5 "epochs". But for the neural network training, I believe the situation is slightly different, as here, unlike in the K-fold cross-validation, it is not guaranteed that all observations are used for training.  

I observed that often after more than 10 epochs, the MSE for the validation set started exceeding the training set error. 

In the end, I use model with 9 epochs.
