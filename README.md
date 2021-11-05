# OpenCV-Backyard-Intruder-Detection

This project uses an OpenCV Background Subtractor MOG2 model
for detecting an intruder in a sequence of images. Though the
model here does a pretty good job of serving the purpose, it isn't
robust and might need a few more modifications.


The intruder detection systems have a wide range of applications and
are widely used. In this project, it was proved that, it's not necessary
to apply heavy object detection algorithms using Deep Learning and some tasks
can be solved with some classical algorithms in computer vision itself.
The approach taken was very simple, read images and create a Background
model based on some history and detect changes and segment the frame using 
watershed. A more in depth explanation of steps followed can be found 
below. 

## Steps Followed

- We start by reading the images one by one from the directory passed as an argument to the
program and perform certain operations on them to give the result of intrusion detection.


- Once we read an image, we perform blur operations and intensity smoothing by converting it
to LAB image and updating the L channel and convert it back to BGR image. This operation
is done in order to smoothen the sharp lighting variation in the frame due to global lighting
changes. Also, since frame are too big for the display we resize each frame with factor defined
by RESIZE_FACTOR to fit it on the screen.


- We find the foreground and background objects in the frame by applying Mixture-of-Gaussians
model available in OpenCV library. It is passed with a certain number of images to build
background model from as history, which can be varied.


- Once we have the foreground and background, we perform distance transform on the foreground
object to calculate distance of each foreground pixel to nearest background pixel and apply
thresholding on it to get fixed foreground pixels.


- Now, we find the markers using connected components method from OpenCV and here we add
1 to every label to make sure background isn’t 0 but 1. For locations which we are unsure
about whether they come in background or foreground, we keep them 0.


- Once we have these markers, we pass the original image and markers to the watershed algorithm
to give us the result of segmentation. While doing this, we need to make sure we pass the same
color marker for all foreground objects detected in the frame.


- Now, we just display the original frame and segmentation results.


- To save a certain frame and its output we can press ’s’.


- Repeat steps ’b’ to ’h’ until we iterate through all the images in the given folder.


## Usage
```shell
python3 intruder_detection.py <path to folder of images>
```
