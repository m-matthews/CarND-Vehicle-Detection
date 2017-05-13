# Vehicle Detection Project

[//]: # (Image References)

[imagesplit1]: ./output_images/split_ShuffleSplit.png "Shuffle Split"
[imagesplit2]: ./output_images/split_GroupShuffleSplit.png "Group Shuffle Split"
[imagehog1]: ./output_images/hog_features.png "HOG Features Checkerboard"
[imagenorm1]: ./output_images/norm_features.png "Normalised Features"
[imagecombined1]: ./output_images/combined_features.png "Combined Features"
[imagesubset1]: ./output_images/simg_0.jpg "Sliding Window Subset (1)"
[imagesubset2]: ./output_images/simg_1.jpg "Sliding Window Subset (2)"
[imagesubset3]: ./output_images/simg_2.jpg "Sliding Window Subset (3)"
[imagetest1]: ./output_images/test_test1.jpg "Test Vehicle Detection (1)"
[imagetest2]: ./output_images/test_test2.jpg "Test Vehicle Detection (2)"
[imagetest3]: ./output_images/test_test3.jpg "Test Vehicle Detection (3)"
[imagetest4]: ./output_images/test_test4.jpg "Test Vehicle Detection (4)"
[imagetest5]: ./output_images/test_test5.jpg "Test Vehicle Detection (5)"
[imagetest6]: ./output_images/test_test6.jpg "Test Vehicle Detection (6)"
[videoout1]: ./output_images/project_video_output.mp4 "Processed Video"
[imagenonveh1]: ./output_images/NonVehicle/nonveh1.jpg "Non Vehicle Image (1)"
[imagenonveh2]: ./output_images/NonVehicle/nonveh2.jpg "Non Vehicle Image (2)"
[imagenonveh3]: ./output_images/NonVehicle/nonveh3.jpg "Non Vehicle Image (3)"
[imagenonveh4]: ./output_images/NonVehicle/nonveh4.jpg "Non Vehicle Image (4)"
[imagegti051]: ./output_images/GTI_Left/005/image0052.png
[imagegti052]: ./output_images/GTI_Left/005/image0053.png
[imagegti053]: ./output_images/GTI_Left/005/image0054.png
[imagegti054]: ./output_images/GTI_Left/005/image0057.png
[imagegti055]: ./output_images/GTI_Left/005/image0058.png
[imagegti056]: ./output_images/GTI_Left/005/image0059.png
[imagegti111]: ./output_images/GTI_Left/011/image0121.png
[imagegti112]: ./output_images/GTI_Left/011/image0126.png
[imagegti113]: ./output_images/GTI_Left/011/image0127.png
[imagegti114]: ./output_images/GTI_Left/011/image0128.png
[imagegti115]: ./output_images/GTI_Left/011/image0129.png
[imagegti116]: ./output_images/GTI_Left/011/image0130.png
[imagegti181]: ./output_images/GTI_Left/018/image0215.png
[imagegti182]: ./output_images/GTI_Left/018/image0216.png
[imagegti183]: ./output_images/GTI_Left/018/image0217.png
[imagegti184]: ./output_images/GTI_Left/018/image0218.png
[imagegti185]: ./output_images/GTI_Left/018/image0219.png
[imagegti186]: ./output_images/GTI_Left/018/image0220.png
[imagegti191]: ./output_images/GTI_Left/019/image0221.png
[imagegti192]: ./output_images/GTI_Left/019/image0222.png
[imagegti193]: ./output_images/GTI_Left/019/image0223.png
[imagegti194]: ./output_images/GTI_Left/019/image0224.png
[imagegti195]: ./output_images/GTI_Left/019/image0225.png
[imagegti196]: ./output_images/GTI_Left/019/image0226.png


## Goals

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Data Preparation

The images used for training the classifier are sourced from the Udacity labeled data for [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

These files include a combination of:

1) [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html).
2) [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).
3) Udacity supplied 'Extras' from the project video.

I also increased the amount of training data due to some images in the video stream producing false positives in early pipeline testing.  This included images at the bridge crossing where the guardrail was detected as a vehicle, although in some situations this was correct as it was identifying a vehicle travelling in the opposite direction.

| Example Vehicles | Example Non-Vehicles |
| ---------------- | -------------------- |
| ![alt text][imagegti051] ![alt text][imagegti111] ![alt text][imagegti181] ![alt text][imagegti191] | ![alt text][imagenonveh1] ![alt text][imagenonveh2] ![alt text][imagenonveh3] ![alt text][imagenonveh4] |

The initial data load (the `trainingdata()` function in [01_Training.py](./01_Training.py)) creates additional vehicle training samples by flipping the `GTI_Left` and `GTI_Right` images to provide additional data from a side-on perspective.  This was done as viewing the project video showed that the vehicles are in a side-on position more than a rear-view position.  This expansion of data is shown in the console when generating the training model.

```
Training Images: Car: 8,792 , Non-Car: 9,295
Training Samples: Car: 10,365 , Non-Car: 9,295
```

The number of sample images for Car and Non-Car are reasonably balanced.

The GTI data includes data in a time-series so that subsequent images can be almost identical.  To ensure the training and test data did not include these same images, manual grouping was performed to move related images into sequentially numbered subfolders.

| Example Subfolder | Files |
| ----------------- | ----- |
| `GTI_Left/005` | ![alt text][imagegti051] ![alt text][imagegti052] ![alt text][imagegti053] ![alt text][imagegti054] ![alt text][imagegti055] ![alt text][imagegti056] |
| `GTI_Left/011` | ![alt text][imagegti111] ![alt text][imagegti112] ![alt text][imagegti113] ![alt text][imagegti114] ![alt text][imagegti115] ![alt text][imagegti116] |
| `GTI_Left/018` | ![alt text][imagegti181] ![alt text][imagegti182] ![alt text][imagegti183] ![alt text][imagegti184] ![alt text][imagegti185] ![alt text][imagegti186] |
| `GTI_Left/019` | ![alt text][imagegti191] ![alt text][imagegti192] ![alt text][imagegti193] ![alt text][imagegti194] ![alt text][imagegti195] ![alt text][imagegti196] |

The names of these subfolders were then used to create a grouping variable (`g`) to be used with `GroupShuffleSplit()`.  An example of using `GroupShuffleSplit()` can be seen in the following table which has a grouping variable corresponding to the color of the image:

| ShuffleSplit() | GroupShuffleSplit() |
| -------------- | ------------------- |
| ![alt text][imagesplit1] | ![alt text][imagesplit2] |

The group variable can be seen to have the correct effect of ensuring the training and test data do not have the same images included.  The sample program [00_ShuffleSplit.py](./00_ShuffleSplit.py) produced these images.


## Features

Three combined feature sets were used to train the classifier.

![alt text][imagecombined1]

The code allows each feature extraction to be derived from individual color spaces using three separate variables:

* `bspace`: Color space to use for spatially binned color.
* `cspace`: Color space to use for color histograms.
* `hspace`: Color space to use for Historgram of Orient Gradients (HOG).

I had originally expected variation could produce improvements, however after experimentation I selected `YUV` as the color space for all feature extraction.


### Historgram of Orient Gradients

The Histogram of Gradients (HOG) perform differently depending on the color space and channel(s) selected.  The start of [01_Training.py](./01_Training.py) exports the following image to enable a review of each channel and corresponding HOG features created:

![alt text][imagehog1]

Based on the above image and experimentation for the Histogram of Gradients I selected the `YUV` color space and selected only channel 0 (`Y`) for training the classifier.  Increasing the number of orientations from 9 to 12 improved the classifier performance.


### Color (Spatial Binning and Histogram)

Various experiments were performed to try different hyperparameters and their impacts.

The binned color size selected was 16x16 as this performed slightly better than the initial 32x32 size.


### Selected Hyperparameters

Experimentation of the hyperparameters resulted in the current selection shown in [01_Training.py](./01_Training.py):

```python
    hp = { # Binned color parameters.
           'bspace': 'YUV',
           'spatial_size': (16, 16),
           # Color histogram parameters.
           'cspace': 'YUV',
           'hist_bins': 32,
           'hist_range': (0, 256),
           # HOG parameters
           'hspace': 'YUV',
           'hchannel': 0,
           'orient': 12,
           'pix_per_cell': 8,
           'cell_per_block': 2
         }
```

Note that at the end of the `train()` method these parameters are saved into [vehicles.yaml](./carnd/vehicles.yaml) for use in the later image and video processing pipeline.


### Feature Normalisation

The feature set is then normalised to enable the model to train correctly.  The following image shows the input and output of the normalisation step:

![alt text][imagenorm1]


## Classifier Training

The initial classifier training was performed with a searching algorithm using `GridSearchCV` (see using the `method="grid"` in `train()` in [01_Training.py](./01_Training.py)).  After a number of iterations and selections, a `linear` SVC was selected with a `C` = 0.001.

```
1.99 Seconds to train SVC(C=0.001)...
Test Accuracy of SVC =  0.9816
```

The output of the training program [01_Training.py](./01_Training.py) includes:

* [vehicles.yaml](./carnd_vehicles.yaml) - Details the final selected hyperparameters.
* vehicles_model.pkl - The model classifier.
* vehicles_scaler.pkl - The X Scaler.

These files enable the subsequent Pipeline program to be able to implement the prediction and correct feature extraction process to match the saved model.


## Sliding Window Search

The `process()` method in the `Vehicles` class performs the sliding window process for detecting vehicles.  This is performed at three different scales.

The selected scales are shown in the creation of the `Vehicles` instance at the start of [02_Pipeline.py](./02_Pipeline.py).  The scale parameters show the scaling factor and the start/stop y coordinate from the input image.

```python
    veh = Vehicles(scales = ((1.0, 402, 498),
                             (1.5, 400, 520),
                             (2, 410, 538)),
```

The following images show the scale and indicate the number of scanned 64x64 window segments.  Note that the first square is thicker to show the scanning window size.

1) Image Scale: 1.0, Y Start: 402, Y Stop: 498.

![alt text][imagesubset1]

2) Image Scale: 1.5, Y Start: 400, Y Stop: 520.

![alt text][imagesubset2]

3) Image Scale: 2.0, Y Start: 410, Y Stop: 538.

![alt text][imagesubset3]


## Heat Map

Where the model predicts a vehicle to be present in the Sliding Window, a rectangle is added to generate a heat map.  This is then thresholded to determine the presence of a vehicle in that region of the image.

This heat map is visualised in all images to enable debugging of the Vehicle Detection functionality by using the `hud=True` options in the creation of the `Vehicle` class instance.


| Heat Map present in the upper right Heads Up Display |
| ---------------------------------------------------- |
| ![alt text][imagetest4]                              |

The threshold for the heatmap involved iterative testing to determine a suitable value based on the number of frames used to smooth the detection.  The final solution (as shown towards the bottom of the `process()` method) was 3 times the number of frames currently in the buffer:

```python
        heat_thresh = int(len(self.rectangles)*3)
        heatmap[heatmap <= heat_thresh] = 0
```

The `scipy.ndimage.measurements.label` function is used to generate the list of distinct vehicles based on the heatmap.  This is then represented onto the image as the vehicle bounding box.


## Merging Vehicle Detection with Lane Line Detection

The classes from [Advanced Lane Line Detection](https://github.com/m-matthews/CarND-Advanced-Lane-Lines/blob/master/writeup.md) Project's [03_Pipeline.py](https://github.com/m-matthews/CarND-Advanced-Lane-Lines/blob/master/03_Pipeline.py) were refactored slightly and moved into the [carnd](./carnd) folder to create a consistent structure.

| Python File                      | Class   | Description                                                                                                     |
| -------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| [camera.py](./carnd/camera.py)   | Camera  | Perform camera calibration and to distort the images to enable detection of straight lane lines.                |
| [lane.py](./carnd/lane.py)       | Lane    | Track the current lane.                                                                                         |
| [line.py](./carnd/line.py)       | Line    | Track a given lane from the current image.  A separate instance is used to track the left and right lane lines. |
| [utils.py](./carnd/utils.py)     |         | Utility functions based on functions developed during the lessons.                                              |
| [vehicle.py](./carnd/vehicle.py) | Vehicle | Detect vehicles in the current images as per the current writeup.                                               |

| Configuration File | Description                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| [camera.yaml](./carnd/camera.yaml) | Camera Calibration configuration used by the Camera class. |
| [perspective_transform.yaml](./carnd/perspective_transform.yaml) | Perspective transformation used by the Lane class to create a 'top-down' view of the road surface. |
| [vehicles.yaml](./carnd/vehicles.yaml) | Hyperparameters used by the vehicle detection training.  The values are saved in this file to enable later pipeline code to ensure it uses consistent parameters. |


## Output Video

The [project video](./output_images/project_video_output.mp4) correctly identifies the vehicles within the input video stream.

Note that oncoming traffic is detected at 0:09 and 0:22, however due to the heat-map thresholding using previous frames to smooth any intermittent detection, this results in the detection of the fast-moving oncoming vehicles drawing squares slightly to the rear of those vehicles.

The video and pipeline also make use of the `Line` and `Lane` classes developed in the previous [Advanced Lane Line Detection](https://github.com/m-matthews/CarND-Advanced-Lane-Lines/blob/master/writeup.md) project.


## Discussion

The pipeline is not currently 'real-time'.  The performance could be improved by using OpenCV as the HOG generator, and alternatively using entirely different techniques for vehicle detection with Neural Networks without using the HOG processing at all.

The classifier currently required some work to remove false positives by including additional 'non-car' examples.  This suggests that an alternative approach should be used for this project (eg: Neural Networks) as otherwise additional environments such as cityscapes etc may cause additional false positive detection.


## Appendix 1: Contents of 'output_images'

The `./output_images` folder contains the following:

* `hog_features.png`: HOG features for each color space.
* `norm_features.png`: Normalised version of the feature vector.
* `simg_*`: Subset images for each layer of scale used in the pipeline.
* `split_*`: Images from [00_ShuffleSplit.py](./00_ShuffleSplit.py) demonstrating the use of `GroupShuffleSplit()`.
* `test_*`: Test images processed from `./test_images` (detailed in **Appendix 2: Test Images**).
* `project_video_output.mp4`: Processed video.


## Appendix 2: Test Images

The following list shows the identification of the lane lines based on the images supplied in the `./test_images` folder:

| Image                   |
|:-----------------------:|
| ![alt text][imagetest1] |
| * Note that the first image includes additional debugging showing 'all' windows where the vehicle prediction = 1. |
| ![alt text][imagetest2] |
| ![alt text][imagetest3] |
| ![alt text][imagetest4] |
| ![alt text][imagetest5] |
| ![alt text][imagetest6] |
