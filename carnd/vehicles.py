# -*- coding: utf-8 -*-
"""
Vehicle Detection Project - Vehicle

Author: Michael Matthews
"""

import numpy as np
import cv2
import time
import yaml
from sklearn import svm
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from scipy.ndimage.measurements import label
from .utils import convert_colors
from .utils import color_hist
from .utils import bin_spatial
from .utils import get_hog_features
from .utils import drawtext

class Vehicles:
    """Class to detect vehicles in an image or video sequence of images.

    This class ...

    Attributes:
        nframes: Number of frames to use in vehicle tracking history.
        hp_yaml: YAML file for model hyperparameters.
        model_pkl: Pickle file for model.
        xscaler_pkl: Pickle file for xscaler.
        scales: List of scales to be used for vehicle detection.
        hud: Is this instance implementing a heads-up-display view in the top-right?
        rectangles: List of rectangles identified as containing a vehicle.
        cspace: Color Histogram - color space.
        spatial_size: Color Histogram - Spatial Size.
        hist_bins: Color Histogram - Bins.
        hist_range: Color Histogram - Bin Range.
        hspace: HOG Features - color space.
        hchannel: HOG Features - channel to use from hspace [0|1|2|"ALL"].
        orient: HOG Features - Orientations.
        pix_per_cell: HOG Features - Pixels per cell.
        cell_per_block: HOG Features - Cells per block.
        svc: Classifier.
        X_scaler: Feature scaler to use prior to classifier.
    """
    
    nframes = 12
    hp_yaml = "./carnd/vehicles.yaml"
    model_pkl = "./carnd/vehicles_model.pkl"
    xscaler_pkl = "./carnd/vehicles_scaler.pkl"

    def __init__(self, hp=None, scales=None, hud=False):
        """Initialise the Vehicle Detection instance.
    
        Args:
            hp: Hyperparameters dictionary.
            scales: Scales to use in Sliding Window vehicle detection.
            hud: Include Heads Up Display (hud) debug of vehicle detection?
        """
        self.scales = scales
        self.hud = hud
        self.rectangles = []

        # Read the YAML model details.
        if hp is None:
            self._readmodel()
        else:
            self._hyperparms(hp)

        return

    def _hyperparms(self, hp):
        """Import a python dictonary set of hyperparameters."""
        self.cspace = hp['cspace']
        self.spatial_size = tuple(hp['spatial_size'])
        self.hist_bins = hp['hist_bins']
        self.hist_range = tuple(hp['hist_range'])
        self.hspace = hp['hspace']
        self.hchannel = hp['hchannel']
        self.orient = hp['orient']
        self.pix_per_cell = hp['pix_per_cell']
        self.cell_per_block = hp['cell_per_block']
        return

    def _readmodel(self):
        """Read the current saved model classifier."""
        with open(self.hp_yaml, "r") as f:
            hp = yaml.load(stream=f)
            self.cspace = hp['cspace']
            self.spatial_size = tuple(hp['spatial_size'])
            self.hist_bins = hp['hist_bins']
            self.hist_range = tuple(hp['hist_range'])
            self.hspace = hp['hspace']
            self.hchannel = hp['hchannel']
            self.orient = hp['orient']
            self.pix_per_cell = hp['pix_per_cell']
            self.cell_per_block = hp['cell_per_block']

        self.svc = joblib.load(self.model_pkl)
        self.X_scaler = joblib.load(self.xscaler_pkl)

        return

    def image_features(self, images):
        """Calculate the features for a list of images.

        Args:
            images: Input images in RGB form.

        Returns:
            Feature lists (normal and scaled).
        """

        # Create a list to append feature vectors to
        self.features = []
    
        # Iterate through the list of images
        for image in images:
            # Append the new feature vector to the features list
            X = self._extract_features(image)
            self.features.append(X)

        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(self.features)

        # Apply the scaler to X
        self.scaled_X = self.X_scaler.transform(self.features)

        # Return list of feature vectors
        return self.features, self.scaled_X
    
    def _extract_features(self, img):
        """Calculate the color and hog features.
    
        Args:
            img: Input images in RGB form.
    
        Returns:
            Feature list.
        """

        # Convert image to the color and hog colorspaces.
        cimg, himg = convert_colors(img, self.cspace, self.hspace)
    
        # Apply bin_spatial() to get spatial color features
        f1 = bin_spatial(cimg, size=self.spatial_size)
        # Apply color_hist() to get color histogram features
        f2 = color_hist(cimg, nbins=self.hist_bins, bins_range=self.hist_range)
    
        # Apply color_hist() to get color histogram features
        if self.hchannel == "ALL":
            f3 = []
            for i in range(3):
                himgc = himg[:,:,i]
                f3.append(get_hog_features(himgc, orient=self.orient,
                                           pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block))
            flist = [f1, f2, f3[0], f3[1], f3[2]]
        else:
            himgc = himg[:,:,self.hchannel]
            f3 = get_hog_features(himgc, orient=self.orient,
                                  pix_per_cell=self.pix_per_cell,
                                  cell_per_block=self.cell_per_block)
            flist = [f1, f2, f3]
    
        # Append the new feature vector to the features list
        return np.concatenate(flist).astype(np.float64)

    def train(self, X, y, g, test_size=0.2,
                    parameters = {'kernel':['linear'], 'C':[1]},
                    method = "simple"):
        """Train a new Vehicle Detection model classifier.
    
        Args:
            X: Input images.
            y: Image flag (1=Vehicle, 0=Non-Vehicle).
            g: Image group.
            test_size: Test size.
            parameters: Parameter list.  A complex set will be used with GridSearchCV().
            method: Method of training.  "simple" will use the current parameters, and "grid" will use GridSearchCV().
    
        Returns:
            Console output showing the details of the created model.
            Pickle and yaml files created containing model, scaler and hyperparameters.
        """

        # Split up data into randomized training and test sets
        if g is None:
            cv = ShuffleSplit(n_splits=1, test_size=test_size,
                              random_state=42)
        else:
            cv = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                   random_state=42)

        if method == "grid":
            print("Grid training with parameters:", parameters)
            svr = svm.SVC(cache_size=500, verbose=1, max_iter=-1)
            clf = GridSearchCV(svr, parameters, cv=cv,
                               n_jobs=4 if self.hchannel=='ALL' else 7)
            t=time.time()
            if g is None:
                clf.fit(X, y)
            else:
                clf.fit(X, y, groups=g)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train SVC...')
            print("Results:", clf.best_params_)

        elif method == "simple":
            print("Simple training.")
            # Use a linear SVC 
            if g is None:
                cvs = cv.split(X, y)
            else:
                cvs = cv.split(X, y, groups=np.array(g))
            
            for train, test in cvs:
                X_train = [X[i] for i in train]
                y_train = [y[i] for i in train]
                X_test = [X[i] for i in test]
                y_test = [y[i] for i in test]
                break
            
            print("Training data: {}, Testing data: {}.".format(len(y_train), len(y_test)))

            # Train a linear SVC (note that multiple C options are possible here
            # even when not using the Grid option).
            for c in parameters['C']:
                clf = LinearSVC(C=c)
                t=time.time()
                clf.fit(X_train, y_train)
                t2 = time.time()
                print(round(t2-t, 2), 'Seconds to train SVC(C={})...'.format(c))
                # Check the score of the SVC
                print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
                # Check the prediction time for a single sample
                t=time.time()
                n_predict = 10
                print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
                print('For these',n_predict, 'labels: ', y_test[0:n_predict])
                t2 = time.time()
                print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
            
        else:
            print("ERROR: Unknown method for 'vehicles.train()'")
            return

        # Export the model details.
        joblib.dump(clf, self.model_pkl)
        joblib.dump(self.X_scaler, self.xscaler_pkl)
        with open(self.hp_yaml, "w") as f:
            yaml.dump({'cspace': self.cspace,
                       'spatial_size': self.spatial_size,
                       'hist_bins': self.hist_bins,
                       'hist_range': self.hist_range,
                       'hspace': self.hspace,
                       'hchannel': self.hchannel,
                       'orient': self.orient,
                       'pix_per_cell': self.pix_per_cell,
                       'cell_per_block': self.cell_per_block}, stream=f)
        return
    
    def reset(self):
        """Reset the vehicle detection over frames for a new image feed."""
        self.rectangles = []
        return

    def process(self, img, debug=False):
        """Process an image into the Vehicle Detection stream.

        Args:
            img: Input images in RGB form.
            debug: Optionally set to output debugging information.

        Returns:
            None.
        """

        heatmap = np.zeros_like(img[:,:,0])

        rectangles = []

        for i, (scale, ystart, ystop) in enumerate(self.scales):
            # Create Color Image and HOG Image.
            cimg, himg = convert_colors(img[ystart:ystop,:,:], self.cspace, self.hspace)
            if debug:
                simg = img[ystart:ystop,:,:].copy()
        
            if scale != 1:
                imshape = cimg.shape
                if debug:
                    simg = cv2.resize(simg, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
                cimg = cv2.resize(cimg, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
                himg = cv2.resize(himg, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

            hchn = []

            if self.hchannel == "ALL":
                for i in range(3):
                    himgc = himg[:,:,i]
                    hchn.append(get_hog_features(himgc, orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                 cell_per_block=self.cell_per_block, 
                                                 feature_vec=False))
            else:
                himgc = himg[:,:,self.hchannel]
                hchn.append(get_hog_features(himgc, orient=self.orient, pix_per_cell=self.pix_per_cell,
                                             cell_per_block=self.cell_per_block,
                                             feature_vec=False))

            # Define blocks and steps as above
            nxblocks = (cimg.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
            nyblocks = (cimg.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 

            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
            
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step

                    # Extract HOG for this patch
                    if self.hchannel == "ALL":
                        hog_feat1 = hchn[0][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                        hog_feat2 = hchn[1][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                        hog_feat3 = hchn[2][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    else:
                        hog_features = hchn[0][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

                    xleft = xpos*self.pix_per_cell
                    ytop = ypos*self.pix_per_cell

                    if debug:
                        cv2.rectangle(simg, (xleft, ytop), (xleft+window, ytop+window),
                                      (0,0,255), 3 if xb == 0 and yb == 0 else 1)

                    # Extract the image patch
                    subimg = cv2.resize(cimg[ytop:ytop+window, xleft:xleft+window], (64,64))

                    # Get color features
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)
                    hist_features = color_hist(subimg, nbins=self.hist_bins, bins_range=self.hist_range)

                    # Scale features and make a prediction
                    test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                    test_prediction = self.svc.predict(test_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        rectangles.append(((xbox_left, ytop_draw+ystart),
                                           (xbox_left+win_draw,ytop_draw+win_draw+ystart)))

            if debug:
                cv2.imwrite("./output_images/simg_{}.jpg".format(i), cv2.cvtColor(simg, cv2.COLOR_BGR2RGB))

        if len(self.rectangles)>= self.nframes:
            self.rectangles.pop(0)
        self.rectangles.append(rectangles)

        # Zero out pixels below the heatmap threshold.
        for i, frame in enumerate(self.rectangles):
            for rect in frame:
                weight = 1 if i<len(self.rectangles)*3//4 else 2
                heatmap[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] += weight

        heat_thresh = int(len(self.rectangles)*3)
        heatmap[heatmap <= heat_thresh] = 0

        self.labels = label(heatmap)

        if self.hud:
            # HUD image will contain a grayscale version of the image with the
            # heat map projection.
            hud_img = cv2.resize(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB),
                                 (0, 0), fx=0.25, fy=0.25)
            hud_heatmap = cv2.resize(heatmap, (0, 0), fx=0.25, fy=0.25)*20
            g_channel = np.zeros(hud_heatmap.shape, dtype=hud_heatmap.dtype)
            b_channel = np.zeros(hud_heatmap.shape, dtype=hud_heatmap.dtype)
            self.hud_img = cv2.addWeighted(hud_img, 1.0, cv2.merge((hud_heatmap, g_channel, b_channel)), 0.4, 0)

        if debug:
            print("Found", self.labels[1], "Cars!")

        return

    def draw(self, image, debug=False):
        """Draw output of the Vehicle Detection stream onto an image.

        Args:
            img: Input images in RGB form.
            debug: Optionally set to output debugging information.

        Returns:
            Original image updated with Vehicle Detection details.
        """

        # In debug mode display all detected rectangles prior to heatmapping.
        if debug:
            for frame in self.rectangles:
                for r in frame:
                    cv2.rectangle(image, (r[0][0], r[0][1]),
                                         (r[1][0], r[1][1]),
                                         (0,0,255), 1) 

        car_count = 0

        for car_number in range(1, self.labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            # In some situations the thresholding was causing a separation of a
            # detection into a main area, and a smaller detached area.  This is
            # likely due to the collection of multiple frames of data.  This code
            # excludes those by removing detections that are vertical in shape.
            xdim = bbox[1][0]-bbox[0][0]
            ydim = bbox[1][1]-bbox[0][1]
            if ydim<xdim*3:
                # Draw the box on the image
                car_count += 1

                cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)
                if self.hud:
                    cv2.rectangle(self.hud_img,
                                  (bbox[0][0]//4, bbox[0][1]//4),
                                  (bbox[1][0]//4, bbox[1][1]//4), (0,0,255), 2)

        # Draw the 'Heads Up Display' if required for this instance.
        if self.hud:
            yoff = 10
            xoff = image.shape[1] - self.hud_img.shape[1] - 10
            image[yoff:yoff+self.hud_img.shape[0], xoff:xoff+self.hud_img.shape[1]] = self.hud_img
            cv2.rectangle(image, (xoff-2, yoff-2), (xoff+self.hud_img.shape[1]+1, yoff+self.hud_img.shape[0]+1),
                          (255, 255, 255), 2)

        drawtext(image, "Vehicles Detected: {}.".format(car_count), 2)

        return image
