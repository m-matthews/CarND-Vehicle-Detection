# -*- coding: utf-8 -*-
"""
Vehicle Detection - Utility functions (based on Lesson content).

Author: Michael Matthews
"""

import numpy as np
import cv2
from skimage.feature import hog

def convert_color(img, colspace='YCrCb'):
    """Convert the colorspace for an image.

    Args:
        img: Input image.
        colspace: Destination color space (assumes input is RGB).

    Returns:
        Converted image.
    """
    if colspace=='RGB':
        return img
    elif colspace=='HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif colspace=='LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif colspace=='YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif colspace=='HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif colspace == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        print("ERROR: Unknown destination color space '" + colspace + "'.")
        return img

def convert_colors(img, cspace, hspace):
    """Convert one image into two separate color spaces.

    Args:
        img: Input image.
        cspace: Destination color space (assumes input is RGB).
        hspace: HOG destination color space (assumes input is RGB).

    Returns:
        Converted image.
    """
    cimg = convert_color(img, cspace)
    himg = convert_color(img, hspace)
    return cimg, himg

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    """Convert an image into hog features for input to model training and prediction.

    Args:
        img: Input image.
        orient: Number of orientations.
        pix_per_cell: Pixels per cell to evaluate.
        cells_per_block: Cell per block.
        vis: Create a visualisation?
        feature_vector: Create a feature vector?

    Returns:
        HOG Features and an optional image (if vis==True).
    """

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    """Convert an image into a flattened feature bin of all channels.

    Args:
        img: Input image.
        size: New size of image prior to binning,

    Returns:
        Single vector with each color channel represented sequentially.
    """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Create a histogram for all colors in the image.

    Args:
        img: Input image.
        nbins: Number of bins,
        bins_range: Low/High range for the value in the bins,

    Returns:
        Single vector with each color channel represented sequentially.
    """
    """Compute the histogram of the color channels separately."""
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def drawtext(image, text, line, size=1.0):
    """Display text on an image in a standard position."""
    cv2.putText(image, text, (10, 35+int(line*35)), cv2.FONT_HERSHEY_SIMPLEX,
                size, (255, 255, 255), thickness=2 if size==1.0 else 1)
    return
