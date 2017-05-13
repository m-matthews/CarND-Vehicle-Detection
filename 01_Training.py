# -*- coding: utf-8 -*-
"""
Vehicle Detection - Model Training

Author: Michael Matthews
"""

#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import itertools
from carnd.utils import convert_color
from carnd.utils import get_hog_features
from carnd.vehicles import Vehicles

TRAIN_TEST_SPLIT = 0.2

# Color Space list used for demonstrating options.
COLOR_SPACES = ('RGB', 'HSV', 'HLS', 'LUV', 'YUV', 'YCrCb')

def sampleimage(chanset):
    """Create a sample blank image with a fixed color.
    
    Args:
        chanset: [0] = Red, [1] = Green, [2] = Blue, [1,2] = Cyan ...
    
    Returns:
        image 64 pixels square.
    """
    image = np.zeros((64,64,3), np.uint8)
    image[:,:,chanset] = 255
    return image
    
def trainingdata(source="small"):
    """Generate list of training data.
    
    Args:
        source: ["small"|"all"]
    
    Returns:
        X(images), y, g(group).
    """
    training_car = 0
    training_noncar = 0
    if source == "small":
        images_car = glob.glob("./training_images/vehicles_smallset/cars*/*.jpeg")
        images_noncar = glob.glob("./training_images/non-vehicles_smallset/notcars*/*.jpeg")
        print("Training Images: Car:", len(images_car), ", Non-Car:", len(images_noncar))
        Ximages = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
                   for file in itertools.chain(images_car, images_noncar)]
        y = np.hstack((np.ones(len(images_car)), np.zeros(len(images_noncar))))
        g = None
        training_car = len(images_car)
        training_noncar = len(images_noncar)

    else:
        images_car = (glob.glob("./training_images/vehicles/GTI_*/*/*.png") +
                      glob.glob("./training_images/vehicles/KITTI_extracted/*.png"))
        images_noncar = (glob.glob("./training_images/non-vehicles/GTI/*.png") +
                         glob.glob("./training_images/non-vehicles/Extras/*.png") +
                         glob.glob("./training_images/non-vehicles/hard_negs/*.jpg"))
        print("Training Images: Car:", len(images_car), ", Non-Car:", len(images_noncar))
        Ximages = []
        y = []
        g = []
        for file in images_car:
            path = file.split("/")
            Ximages.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
            y.append(1)
            # Group is "GTI_<type>.<file-group>".
            g.append(".".join(path[-3:-1]))
            training_car = training_car + 1

            # Increase the size of the training data by flipping Left and Right images.
            if path[-3] in ("GTI_Left", "GTI_Right"):
                Ximages.append(cv2.flip(Ximages[-1],1))
                y.append(1)
                # Group is "GTI_<type>.<file-group>".
                g.append(".".join(path[-3:-1]))
                training_car = training_car + 1

        gid = 0
        for file in images_noncar:
            path = file.split("/")
            Ximages.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
            y.append(0)
            # Group is "GTI_<type>.<file-group>".
            g.append("Noncar."+str(gid))
            gid = gid + 1
            training_noncar = training_noncar + 1

    print("Training Samples: Car:", training_car, ", Non-Car:", training_noncar)

    return (Ximages, y, g)

if __name__ == '__main__':
    # Read training image filenames.
    Ximages, y, g = trainingdata("all")
    #Ximages, y, g = trainingdata("small")

    # Generate HOG features colorspace examples.
    fig, ax = plt.subplots(len(COLOR_SPACES), 7, figsize=(6,6))
    fig.suptitle('HOG Display by Color Space')
    for i, hspace in enumerate(COLOR_SPACES):
        rgbimage = Ximages[0].copy()
        image = convert_color(rgbimage, hspace)
        ax[i][0].imshow(rgbimage)
        ax[i][0].axis('off')
        for channel in (0, 1, 2):
            # Plot the examples
            cimage = image[:,:,channel]
            features, himage = get_hog_features(cimage, 9, 8, 2, vis=True)
            ax[i][1+channel*2].imshow(cimage, cmap='gray')
            ax[i][1+channel*2].axis('off')
            ax[i][1+channel*2].set_title(hspace + '[' + str(channel) + ']')
            ax[i][2+channel*2].imshow(himage, cmap='gray')
            ax[i][2+channel*2].axis('off')
    plt.savefig("./output_images/hog_features.png")

    hp = { # Color parameters.
           'cspace': 'YUV',
           'spatial_size': (32, 32),
           'hist_bins': 32,
           'hist_range': (0, 256),
           # HOG parameters
           'hspace': 'YUV',
           'hchannel': 0,
           'orient': 12,
           'pix_per_cell': 8,
           'cell_per_block': 2
         }

    v = Vehicles(hp)

    # Generate training features.
    X, scaled_X = v.image_features(Ximages)

    # Plot an example of raw and scaled features
    car_ind = 42
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(Ximages[car_ind])
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.savefig("./output_images/norm_features.png")

    # Example code for training with GridSearchCV().
    #v.train(scaled_X, y, g, test_size=TRAIN_TEST_SPLIT,
    #parameters = {'kernel':['linear','rbf'], 'C':[0.01, 0.1, 1, 10]},
    #method = "grid")

    # Train with the final selected parameters.
    v.train(scaled_X, y, g, test_size=TRAIN_TEST_SPLIT,
            parameters = {'kernel':['linear'], 'C':[0.001]},
            method = "simple")
