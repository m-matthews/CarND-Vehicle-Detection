# -*- coding: utf-8 -*-
"""
Advanced Lane Line Project - Camera

Author: Michael Matthews
"""

import os
import glob
import numpy as np
import cv2
import yaml

class Camera:
    """Class to perform camera calibration.

    This class accepts binary image(s) and detects pixels likely to represent
    the lane line ('left'|'right').  The resulting pixels are fitted with a
    polynomial expression.
    Subsequent images are fitted, taking into account the previous images to
    improve lane detection.

    Attributes:
        mtx: Camera Calibration Matrix.
        dist: Camera Calibation Distortion.
        yamlfile: YAML configuration file.
    """
    
    yamlfile = "./carnd/camera.yaml"

    def __init__(self):
        """Undistort the image based on the camera parameters."""
        if os.path.isfile(self.yamlfile):
            with open(self.yamlfile, "r") as f:
                camcal = yaml.load(stream=f)
                self.mtx = np.array(camcal['mtx'])
                self.dist = np.array(camcal['dist'])
        else:
            self.mtx = None
            self.dist = None
            print("Note: '{}' does not exist.".format(self.yamlfile))

        return
    
    def process(self, image):
        """Undistort the image based on the camera parameters."""
        if self.mtx is None:
            return image
        else:
            return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def calibrate(self, filepath="camera_cal/calibration*.jpg",
                  chessboard_x=9, chessboard_y=6):
        objpoints = [] # 3D points in real world space.
        imgpoints = [] # 2D points in image plane.

        # Generate the list of camera calibration filenames.
        imagefiles = sorted(glob.glob(filepath))
    
        # Prepare object point structures.
        objp = np.zeros((chessboard_y*chessboard_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2) # X, Y coordinates.
    
        # Loop through the list of calibration file names.
        for imagefile in imagefiles:
            img = cv2.imread(imagefile)
    
            # Convert to grayscale.
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            # Detect chessboard corners.
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_x, chessboard_y), None)
    
            # If corners are detected, add to the lists of object and image points.
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
    
                # Preview detection is working correctly.
                img = cv2.drawChessboardCorners(img, (chessboard_x, chessboard_y), corners, ret)
                cv2.imwrite(os.path.join("output_images", "ccal_" + imagefile.split("/")[-1]), img)
            else:
                print("Unable to find chessboard corners for '{}'".format(imagefile))
    
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
        # Save the camera calibration details to a YAML format for later processes.
        with open(self.yamlfile, "w") as f:
            yaml.dump({'ret': ret, 'mtx': self.mtx.tolist(), 'dist': self.dist.tolist()}, stream=f)

        return
