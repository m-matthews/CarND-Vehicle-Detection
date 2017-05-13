# -*- coding: utf-8 -*-
"""
Advanced Lane Line Project - Pipeline

Author: Michael Matthews
"""

import numpy as np
import cv2
import yaml
from .line import Line
from .utils import drawtext

class Lane:
    """Class to contain lane details of the left and right line.

    This class accepts image and uses the Line class to track the position of
    the left and right line making up the current lane on the road.

    Attributes:
        ksize: Kernel size to use in Sobel operations.
        niterations: Number of image processing iterations to keep for
                     determining the average road radius to remove noise.
        firstscan: Detect if this is the first image scanning operation of a
                   movie sequence.
        iterfits: List of curve fit polynomial from each iteration.
        left_line: Instance of the Line class for the left lane line.
        right_line: Instance of the Line class for the right lane line.
        mtx: Camera calibration details.
        dist: Camera calibration details.
        M: Perspective Transformation details.
        xm_per_pix: meters per pixel in x dimension
        ym_per_pix: meters per pixel in y dimension
        debug: Debugging status.
    """
    niterations = 12
        
    # Conversions in x and y from pixels space to meters
    # Based on file: output_images/perspective_transform_out.jpg:
    #   The gap between line centres (3.7m) is ~750 pixels.
    #   The white line (3m) is ~100 pixels high.
    xm_per_pix = 3.7/750
    ym_per_pix = 3/100

    def __init__(self, ksize=3, hud=False, debug=False):
        """Initialise the Line instance.
    
        Args:
            colspace: Color space for input images.
            ksize: Kernal Size to use in Sobel operations.
            hud: Include Heads Up Display (hud) debug of lane finding?
            debug: Flag to save intermediate processing images to disk.
        """

        # Sobel kernel size
        self.ksize = ksize
        # Is this the initial scan?
        self.firstscan = True
        # List of centre line fit polynomials.
        self.iterfits = []
        # Keep debugging status.
        self.hud = hud
        self.debug = debug

        # Import Perspective Transform values created in 02_Perspective_Transform.py.
        with open("./carnd/perspective_transform.yaml", "r") as f:
            ydata = yaml.load(stream=f)
            self.M = np.array(ydata['M'])

        self.left_line = Line("left")
        self.right_line = Line("right")

        return

    def show(self, image, winname="image", binary=False):
        """Display the image in a window for debugging purposes."""
        if binary:
            image=image*255
        cv2.namedWindow(winname)
        cv2.imshow(winname, image)
        cv2.waitKey()
        return

    def _binarythreshold(self, image, thresh):
        """Perform thresholding limits on a binary image."""
        binary_output = np.zeros_like(image)
        binary_output[(image >= thresh[0]) & (image <= thresh[1])] = 1
        return binary_output

    def _abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Calculate the directional threshold gradient.
    
        Args:
            image: Input grayscale image.
            orient: Orientation for scan ('x'|'y').
            sobel_kernel: Kernel size for Sobel function.
            thresh: Threshold in form (min, max).
    
        Returns:
            Binary image based on supplied threshold.
        """
        sobel = cv2.Sobel(image, cv2.CV_64F, 1 if orient=="x" else 0,
                                             0 if orient=="x" else 1,
                                             ksize = sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return self._binarythreshold(scaled_sobel, thresh)
    
    def _mag_thresh(self, image, sobel_kernel=3, thresh=(0, 255)):
        """Calculate the gradient magnitude.
    
        Args:
            image: Input grayscale image.
            sobel_kernel: Kernel size for Sobel function.
            thresh: Threshold in form (min, max).
    
        Returns:
            Binary image based on supplied threshold.
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return self._binarythreshold(scaled_sobel, thresh)
    
    def _dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        """Calculate the gradient direction.
    
        Args:
            image: Input grayscale image.
            sobel_kernel: Kernel size for Sobel function.
            thresh: Threshold in form (min, max).
    
        Returns:
            Binary image based on supplied threshold.
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
        return self._binarythreshold(dir_sobel, thresh)

    def _combined_threshold(self, image, x_thresh, y_thresh, m_thresh, d_thresh):
        """Calculate the combined threshold.
    
        Args:
            image: Input grayscale image.
            x_thresh: Threshold to apply to x directional gradient.
            y_thresh: Threshold to apply to y directional gradient.
            m_thresh: Threshold to apply to magnitude gradient.
            d_thresh: Threshold to apply to gradient direction.
    
        Returns:
            Binary image from combined thresholds.
        """
        gradx = self._abs_sobel_thresh(image, orient='x', sobel_kernel=self.ksize,
                                       thresh=x_thresh)
        grady = self._abs_sobel_thresh(image, orient='y', sobel_kernel=self.ksize,
                                       thresh=y_thresh)
        mag_binary = self._mag_thresh(image, sobel_kernel=self.ksize,
                                      thresh=m_thresh)
        dir_binary = self._dir_threshold(image, sobel_kernel=self.ksize,
                                         thresh=d_thresh)

        combined_binary = np.zeros_like(dir_binary)
        combined_binary[((gradx == 1) & (grady == 1)) |
                        ((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined_binary

    def _thresholds(self, image):
        """Calculate thresholds for all required channels.
    
        Args:
            image: Input grayscale image.
    
        Returns:
            Binary image from combined thresholds.
        """

        # Convert to HLS color space and separate the S channel.
        s_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2]
        # The Red channel is a suitable alternative to gray.
        r_channel = image[:,:,0]

        s_combined_binary=self._combined_threshold(s_channel, x_thresh=(100, 255),
                                                              y_thresh=(100, 255),
                                                              m_thresh=(30, 100),
                                                              d_thresh=(0.7, 1.3))
        r_combined_binary=self._combined_threshold(r_channel, x_thresh=(20, 100),
                                                              y_thresh=(100, 200),
                                                              m_thresh=(60, 150),
                                                              d_thresh=(0.7, 1.3))

        # Initial scan only uses the s channel.  After this the combination
        # with red is suitable to continue lane line detection.
        if self.firstscan == True:
            combined_binary = s_combined_binary
            self.firstscan = False
        else:
            combined_binary = np.zeros_like(s_combined_binary)
            combined_binary[((s_combined_binary == 1) | (r_combined_binary == 1))] = 1

        # Export thresholding images if the debug flag is set.
        if self.debug:
            cv2.imwrite("./output_images/threshold_s_channel.jpg", s_channel)
            cv2.imwrite("./output_images/threshold_r_channel.jpg", r_channel)
            cv2.imwrite("./output_images/threshold_s_binary.jpg", s_combined_binary*255)
            cv2.imwrite("./output_images/threshold_r_binary.jpg", r_combined_binary*255)
        
        return combined_binary

    def _findlane(self, binary, image):
        """Using the supplied images determine the left and right lane lines.
    
        Args:
            binary: Binary image of potential lane line pixels.
            image: Original image frame (distorted).
    
        Returns:
            Updated image for display with lane markings and additional
            information.
        """

        if self.hud:
            # HUD image will contain the top-down display with line detection
            # information.  It will be 1/16 scale in the top-right of the final
            # image projection.
            hud_img = cv2.warpPerspective(image, self.M,
                                          (image.shape[1],image.shape[0]),
                                          flags=cv2.INTER_LINEAR)
            hud_img = cv2.resize(hud_img, (0, 0), fx=0.25, fy=0.25)
            hud_overlay_img = np.zeros_like(hud_img)
        else:
            hud_overlay_img = None

        # Overlay image is used to project the detected lanes back onto the
        # original image/road surface.
        overlay_img = np.zeros_like(image)

        # Process the new input image with each line.
        self.left_line.process(binary, overlay_img, hud_overlay_img)
        self.right_line.process(binary, overlay_img, hud_overlay_img)

        # Add the final overlay polygon using the output of both line detections.
        overlay_poly = []
        for y in range(0, overlay_img.shape[0]+1, 36):
            overlay_poly.append((self.left_line.current_fit[0]*y**2 +
                                 self.left_line.current_fit[1]*y +
                                 self.left_line.current_fit[2], y))
        for y in range(overlay_img.shape[0], -1, -36):
            overlay_poly.append((self.right_line.current_fit[0]*y**2 +
                                 self.right_line.current_fit[1]*y +
                                 self.right_line.current_fit[2], y))

        cv2.fillPoly(overlay_img, [np.array(overlay_poly, dtype=np.int32)], (0, 255, 0))
        
        # Add the average of the two Lines into the current fit.
        self.iterfits.append(np.average([self.left_line.current_fit,
                                         self.right_line.current_fit], axis=0))
        if len(self.iterfits)>self.niterations:
            self.iterfits.pop(0)

        # Create average fit of the all retained iterations.
        avgfit = np.average(self.iterfits, axis=0)
        ploty = np.linspace(0, overlay_img.shape[0]-1, overlay_img.shape[0] )
        plotx = avgfit[0]*ploty**2 + avgfit[1]*ploty + avgfit[2]
        # Restrict the points to within the screen.
        plotx = np.maximum(1, np.minimum(overlay_img.shape[1]-2, plotx))

        overlay_img[np.array(ploty, dtype=np.int32),
                    np.array(plotx-1, dtype=np.int32)] = [0, 0, 0]
        overlay_img[np.array(ploty, dtype=np.int32),
                    np.array(plotx, dtype=np.int32)] = [0, 0, 0]
        overlay_img[np.array(ploty, dtype=np.int32),
                    np.array(plotx+1, dtype=np.int32)] = [0, 0, 0]

        # Calculate the new radii of curvature in meters.
        self.curverad = ((1 + (2*avgfit[0]*np.max(ploty)*self.ym_per_pix + avgfit[1])**2)**1.5) / np.absolute(2*avgfit[0])

        # Warp the overlay image back into the original perspective.        
        self.overlay_img = cv2.warpPerspective(overlay_img, self.M,
                                               (overlay_img.shape[1],overlay_img.shape[0]),
                                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

        # Build the HUD image for the top right corner of the display.
        if self.hud:
            self.hud_img = cv2.addWeighted(hud_img, 1.0, hud_overlay_img, 0.4, 0)
        
        return image

    def reset(self):
        """Completely reset the Lane detection process."""
        self.firstscan = True
        self.iterfits = []
        self.left_line.reset()
        self.right_line.reset()
        return

    def process(self, image):
        """Process a given image to find the left and right lane lines.
    
        Args:
            image: Original image frame.
    
        Returns:
            Updated image for display with lane markings and additional
            information.
        """

        # Perform the thresholding.
        bimage = self._thresholds(image)

        # Flatten the binary image to appear as a top-down view.
        bwarped = cv2.warpPerspective(bimage, self.M,
                                      (bimage.shape[1],bimage.shape[0]),
                                      flags=cv2.INTER_LINEAR)
        
        out_img = self._findlane(bwarped, image)

        # Ensure debugging images are only exported once.
        self.debug = False

        return out_img

    def draw(self, image):

        image = cv2.addWeighted(image, 1.0, self.overlay_img, 0.4, 0)

        if self.hud:
            yoff = 10
            xoff = image.shape[1] - self.hud_img.shape[1] - 10
            image[yoff:yoff+self.hud_img.shape[0], xoff:xoff+self.hud_img.shape[1]] = self.hud_img
            cv2.rectangle(image, (xoff-2, yoff-2), (xoff+self.hud_img.shape[1]+1, yoff+self.hud_img.shape[0]+1),
                          (255, 255, 255), 2)

        # Display curve radius on the output image.
        drawtext(image, "Curve Radius: " +
                       ("Straight." if self.curverad>6000 else "{:.0f}m.".format(self.curverad)), 0)
        
        # Display vehicle left/right offset on the output image.
        veh_offset = (self.left_line.base_offset*self.xm_per_pix +
                      self.right_line.base_offset*self.xm_per_pix)/2
        if abs(veh_offset)<0.02:
            drawtext(image, "Vehicle: Centred.", 1)
        else:
            drawtext(image, "Vehicle: {:.2f}m {} of centre.".format(abs(veh_offset), "right" if veh_offset<0 else "left"), 1)

        return image
