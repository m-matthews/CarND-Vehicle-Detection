# -*- coding: utf-8 -*-
"""
Advanced Lane Line Project - Pipeline

Author: Michael Matthews
"""

import numpy as np
import cv2

class Line:
    """Class to contain details of a given line.

    This class accepts binary image(s) and detects pixels likely to represent
    the lane line ('left'|'right').  The resulting pixels are fitted with a
    polynomial expression.
    Subsequent images are fitted, taking into account the previous images to
    improve lane detection.

    Attributes:
        side: String defining the lane side ('left'|'right').
        nwindows: Number of sliding windows for initial line detection process.
        margin: +/- number of x-pixels to scan in each window.
        minpix: Number of pixels required to recenter window
        niterations: Number of image processing iterations to keep.
        color: Color used to represent this 'side' in displays.
        detected: Was the line detected in the last iteration?
        base: x-pixel location of the base of the line closest to the vehicle.
        base_offset: offset of base from the centre of the image.
        current_fit: polynomial coefficients for the most recent fit.
        iterx: x coordinates detected to be part of the line.  One list for
               each iteration (eg: [[iter5-pts], [iter4-pts], ...])
        itery: y coordinates detected to be part of the line.  One list for
               each iteration (eg: [[iter5-pts], [iter4-pts], ...])
        curverad: Current estimated curve radius.
    """

    nwindows = 9
    margin = 100
    minpix = 50
    niterations = 5

    def __init__(self, side):
        """Initialises the Line class with the 'side' to be processed."""
        self.side = side
        self.color = [255, 0, 0] if side == 'left' else [0, 0, 255]

        self.detected = False
        self.base = None
        self.base_offset = None
        self.current_fit = [np.array([False])]
        self.iterx = []
        self.itery = []
        self.curverad = None 

        return

    def reset(self):
        """
        Called from Lane if the lines are not valid and the search should
        begin again with the windowing functions.
        """
        self.detected = False
        self.iterx = []
        self.itery = []
        return

    def process(self, binary, overlay_img, hud_img):
        """Process the binary image to determine the line position.
    
        The binary image is examined to determine the line position and
        polynomial representation.  The result is displayed onto the 
        overlay_img (which will be flattened back onto the road surface) and
        also the hud_img which will be displayed in the top-right as a
        representation of this line processing.
    
        Args:
            binary: The binary image pre-processed by the Lane class.
            overlay_img: Image to be updated with the relevant points used to
                         detect this line.
            hud_img: Image to be displayed in top-right corner to show inner
                     working of this line detection process.
    
        Returns:
            None.
        """
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Does the initial line base need to be found?
        if self.detected == False:
            window_scan = True
            # Take a histogram of the bottom third of the image
            histogram = np.sum(binary[binary.shape[0]//3:,:], axis=0)

            # Find the peak of the histogram which will be the starting point
            # for the lane line.
            midpoint = np.int(histogram.shape[0]/2)
            if self.side == 'left':
                x_current = np.argmax(histogram[:midpoint])
            else:
                x_current = np.argmax(histogram[midpoint:]) + midpoint
        
            # Set height of windows.
            window_height = np.int(binary.shape[0]/self.nwindows)

            # Create empty lists to receive lane pixel indices
            lane_inds = []
            
            # Step through the windows one by one
            for window in range(self.nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary.shape[0] - (window+1)*window_height
                win_y_high = binary.shape[0] - window*window_height
                win_x_low = x_current - self.margin
                win_x_high = x_current + self.margin

                # Draw the windows on the visualization image
                if hud_img is not None:
                    cv2.rectangle(hud_img,(win_x_low//4,win_y_low//4),
                                          (win_x_high//4,win_y_high//4),
                                          (0,255,0), 1) 
                
                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                # Append these indices to the lists
                lane_inds.append(good_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds) > self.minpix:
                    x_current = np.int(np.mean(nonzerox[good_inds]))
            
            # Concatenate the arrays of indices
            lane_inds = np.concatenate(lane_inds)
            
            # The initial line detection is not required for future calls.
            self.detected = True
        else:
            window_scan = False
            # Assume you now have a new warped binary image 
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            lane_inds = ((nonzerox > (self.current_fit[0]*(nonzeroy**2) +
                                      self.current_fit[1]*nonzeroy +
                                      self.current_fit[2] - self.margin)) &
                         (nonzerox < (self.current_fit[0]*(nonzeroy**2) +
                                      self.current_fit[1]*nonzeroy +
                                      self.current_fit[2] + self.margin)))

        # Remove the oldest iteration if the required limit is reached.
        if len(self.iterx)>=self.niterations:
            self.iterx.pop(0)
            self.itery.pop(0)
            
        # Extract line pixel positions and update the latest iteration values.
        self.iterx.append(nonzerox[lane_inds])
        self.itery.append(nonzeroy[lane_inds])

        # Concatenate all of the points for all iterations into a single list.
        x = np.concatenate(self.iterx)
        y = np.concatenate(self.itery)
        # Build a list of weights based on the iterations.  This is 1.0 for
        # the latest, and 0.8, 0.8, 0.4 and 0.2 for previous iterations.
        w = np.concatenate([np.repeat((i+1)/len(self.iterx), len(self.iterx[i]))
                            for i in range(len(self.iterx))])
        
        # Fit a second order polynomial to the points, using the weighted values.
        self.current_fit = np.polyfit(y, x, 2, w=w)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0] )
        current_fitx = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
        # Restrict the points to within the screen.
        current_fitx = np.maximum(0, np.minimum(binary.shape[1]-1, current_fitx))

        # Determine the base of the line, and offset to centre.
        self.base = current_fitx[-1]
        self.base_offset = self.base - binary.shape[1]//2

        if window_scan == False:
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            line_window1 = np.array([np.transpose(np.vstack([current_fitx-self.margin, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([current_fitx+self.margin, ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            
            # Draw the lane onto the warped HUD image.
            if hud_img is not None:
                cv2.fillPoly(hud_img, np.int_([line_pts])//4, (0, 255, 0))

        if hud_img is not None:
            hud_img[nonzeroy[lane_inds]//4, nonzerox[lane_inds]//4] = self.color
            for i in range(0, len(self.iterx)-1):
                fadecolor = np.maximum(self.color,int(255/self.niterations*(self.niterations-i-1)))
                hud_img[self.itery[i]//4, self.iterx[i]//4] = fadecolor

        overlay_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = self.color

        if hud_img is not None:
            # Draw the contributing points in the color selected for this line.
            hud_img[nonzeroy[lane_inds]//4, nonzerox[lane_inds]//4] = self.color
            hud_img[nonzeroy[lane_inds]//4, nonzerox[lane_inds]//4] = self.color
    
            # Draw the polynomial in yellow.
            hud_img[np.array(ploty//4, dtype=np.int32),
                    np.array(current_fitx//4, dtype=np.int32)] = [255, 255, 0]
        
        return

