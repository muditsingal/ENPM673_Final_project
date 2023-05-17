# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import numpy as np

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline_left(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def gstreamer_pipeline_right(
    sensor_id=1,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    window_title_left = "CSI Camera left"
    window_title_right = "CSI Camera right"
    # Apply identity kernel
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
     
    # Apply blurring kernel
    kernel2 = np.ones((3, 3), np.float32) / 9
    #img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
     
    
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline_left(flip_method=0))
    video_capture_left = cv2.VideoCapture(gstreamer_pipeline_left(flip_method=2), cv2.CAP_GSTREAMER)
    video_capture_right = cv2.VideoCapture(gstreamer_pipeline_right(flip_method=2), cv2.CAP_GSTREAMER)
    if video_capture_left.isOpened() and video_capture_right.isOpened():
        try:
            window_handle_left = cv2.namedWindow(window_title_left, cv2.WINDOW_AUTOSIZE)
            window_handle_right = cv2.namedWindow(window_title_right, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val_left, frame_left = video_capture_left.read()
                ret_val_right, frame_right = video_capture_right.read()
                frame_left = cv2.flip(frame_left, 1)                
                frame_right = cv2.flip(frame_right, 1)
                identity_left = cv2.filter2D(src=frame_left, ddepth=-1, kernel=kernel2)
                identity_right = cv2.filter2D(src=frame_right, ddepth=-1, kernel=kernel2)

                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title_left, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title_left, frame_left)
                    #cv2.imshow(window_title_right, frame_right)
                    cv2.imshow("Filtered left image", identity_left)
                    #cv2.imshow("Filtered right image", identity_right)

                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    cv2.imwrite("Left_img1.jpg", frame_left)
                    cv2.imwrite("Right_img1.jpg", frame_right)
                    break
        finally:
            video_capture_left.release()
            video_capture_right.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
