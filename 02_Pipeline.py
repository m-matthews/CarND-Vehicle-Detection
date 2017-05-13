# -*- coding: utf-8 -*-
"""
Vehicle Detection - Pipeline

Author: Michael Matthews
"""

import os
import cv2
import glob
from moviepy.editor import VideoFileClip

from carnd.lane import Lane
from carnd.vehicles import Vehicles
from carnd.camera import Camera

if __name__ == '__main__':
    # Vehicle Detection instance (including Heads Up Display (hud)).
    # Scales for sliding window are specified here.
    veh = Vehicles(scales = ((1.0, 402, 498),
                             (1.5, 400, 520),
                             (2, 410, 538)),
                   hud=True)

    # Camera calibration instance required for Lane finding.
    camera = Camera()
    # Lane finding (without Heads Up Display (hud)).
    lane = Lane(hud=False)

    # Loop through test images.    
    imagefiles = sorted(glob.glob("./test_images/*.jpg"))
    for imagefile in imagefiles:
        print("Processing", imagefile)
        # Read image with openCV converting to RGB standard.
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
        # Calibrate image.
        image = camera.process(image)
        # Debug flag set for first image (will display all detection rectangles).
        debug = (imagefile == imagefiles[0])
        # Detect vehicles.
        veh.process(image, debug=debug)
        # Detect lane lines.
        lane.process(image)
        # Draw vehicle detections.
        image = veh.draw(image, debug=debug)
        # Draw lane line detection.
        image = lane.draw(image)
        # Export image.
        cv2.imwrite(os.path.join("./output_images", "test_" + imagefile.split("/")[-1]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # Reset instances as subsequent images are unrelated (not a video stream).
        veh.reset()
        lane.reset()
    
    def process_image(image):
        # Calibrate image.
        image = camera.process(image)
        # Detect vehicles and lane lines.
        veh.process(image)
        lane.process(image)
        # Draw vehicle and lane line detections.
        image = veh.draw(image)
        image = lane.draw(image)
        return image
    
    videofile = 'project_video'
    # Process the entire video file.
    clip = VideoFileClip(videofile + '.mp4')
    # Test the bridge crossing 5 seconds of the movie.
    #clip = VideoFileClip(videofile + '.mp4').subclip((0,20), (0,25))
    # Test the woodlands to the right.
    #clip = VideoFileClip(videofile + '.mp4').subclip((0,13), (0,17))
    project_clip = clip.fl_image(process_image)
    project_clip.write_videofile('output_images/' + videofile + '_output.mp4', audio=False)
    # Write the individual video frames as images for testing.
    #clip.write_images_sequence('output_images/video/' + videofile + '_output_%d.jpg')
