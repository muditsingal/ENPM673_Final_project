import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#import tqdm
#import open3d as o3d
#import glob

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

def getMatches(image1, image2, number, detector):
    """
        Initialize a new Contact object.

        Args:
            image1 (cv::Mat): The first image
            image2 (cv::Mat): The second image
            number (int): Number of matches to consider

        Returns:
            points1 (numpy array, int32): Feature Points for first image
            points2 (numpy array, int32): Feature Points for second image

    """
    #sift = cv.SIFT_create()
    #sift = cv.ORB_create()

    keypoints1, descriptors1 = detector.detectAndCompute(image1,None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2,None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key = lambda x:x.distance)
  
    print("No. of matching features found =",len(matches))
    points1 = []
    points2 = []

    for i in matches[:number]:
        x1, y1 = keypoints1[i.queryIdx].pt
        x2, y2 = keypoints2[i.trainIdx].pt
        points1.append([x1, y1])
        points2.append([x2, y2])

    image1_copy = image1.copy()
    image2_copy = image2.copy()

    image1_features = cv.drawKeypoints(image1, keypoints1, image1_copy)
    image2_features = cv.drawKeypoints(image2, keypoints2, image2_copy)

    draw_images = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches[:number], image2_copy, flags=2)

    #cv.imshow("image1_features", image1_features)
    #cv.waitKey(0)

    #cv.imshow("image2_features", image2_features)
    #cv.waitKey(0)

    #cv.imshow("Keypoint matches", draw_images)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return np.array(points1, dtype = np.int32), np.array(points2, dtype = np.int32)


def getDispartiy(image1, image2, h, w):

    #disparity = np.zeros((h,w), np.uint8)
    #disparity = np.zeros((h,w), np.float32)
    image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    #stereo = cv.StereoSGBM_create(1, 128, 3, speckleRange=1)
    stereo = cv.StereoSGBM_create(1, 64, 11, speckleRange=1)
    #stereo = cv.StereoBM_create(numDisparities=64, blockSize=7)
                                # speckleWindowSize=50)
    disparity = stereo.compute(image1_gray, image2_gray)
    disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    print(np.max(disparity), np.min(disparity))

    #disparity = np.float32(disparity/4.01)
    #disparity = disparity.astype(np.uint8)
    #print(np.max(disparity), np.min(disparity))


    return disparity




def main():
    # pcds= []
    window_title_left = "CSI Camera left"
    window_title_right = "CSI Camera right"
    filter_img = False
    dil_kernel = np.ones((3, 3), np.uint8)
    #kernel2 = np.ones((3, 3), np.float32) / 9
    j = 0
    kernel2 = np.ones((7, 7), np.float32) / 49

    '''
    intrinsic_matrix = np.array([[1552.58610, 0, 768.938827],
                                 [0, 1552.57499, 1006.09656],
                                 [0, 0, 1]])

    '''
    intrinsic_matrix = np.array([[1337.89758, 0, 440.149059],
                                 [0, 1338.17020, 299.853520],
                                 [0, 0, 1]])
    
    
    ## TODO: Create a function for the code below 
    # Point Cloud Generation from disparity(depth) and RGB Image
    #image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

    #disparity = np.float32(disparity/16) # As mentioned in documentation (Required step)
    orb_detector = cv.ORB_create()

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline_left(flip_method=0))
    video_capture_left = cv.VideoCapture(gstreamer_pipeline_left(flip_method=2), cv.CAP_GSTREAMER)
    video_capture_right = cv.VideoCapture(gstreamer_pipeline_right(flip_method=2), cv.CAP_GSTREAMER)
    if video_capture_left.isOpened() and video_capture_right.isOpened():
        try:
            window_handle_left = cv.namedWindow(window_title_left, cv.WINDOW_AUTOSIZE)
            window_handle_right = cv.namedWindow(window_title_right, cv.WINDOW_AUTOSIZE)
            while True:
                while j < 20:
                    ret_val_left, frame_left = video_capture_left.read()
                    ret_val_right, frame_right = video_capture_right.read()
                    j += 1

                ret_val_left, frame_left = video_capture_left.read()
                ret_val_right, frame_right = video_capture_right.read()
                #frame_left = cv.flip(frame_left, 1)                
                #frame_right = cv.flip(frame_right, 1)
                if filter_img == True:
                    frame_left = cv.filter2D(src=frame_left, ddepth=-1, kernel=kernel2)
                    frame_right = cv.filter2D(src=frame_right, ddepth=-1, kernel=kernel2)
                points1, points2 = getMatches(frame_left, frame_right, 1000, detector=orb_detector)

                fundamental_matrix, _ = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 0.5, 0.99)

                # essential_matrix = cv.findEssentialMat(points1, points2, intrinsic_matrix, cv.RANSAC, 0.99, 0.1)

                h1, w1 = frame_left.shape[:2]

                _, H1, H2 = cv.stereoRectifyUncalibrated(points1, points2, fundamental_matrix, imgSize=(w1, h1))
                
                #print("\nH1 =", H1) 
                #print("\nH2 =", H2)

                disparity = getDispartiy(frame_left, frame_right, frame_left.shape[0], frame_left.shape[1])
                disparity = cv.dilate(disparity, dil_kernel, iterations=1)
                disparity = cv.filter2D(src=disparity, ddepth=-1, kernel=kernel2)
                disparity = cv.filter2D(src=disparity, ddepth=-1, kernel=kernel2)
                cv.imshow("disparity map obtained", disparity)
                #plt.imshow(disparity, cmap='jet', interpolation='gaussian')
                #plt.show()

                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv.getWindowProperty(window_title_left, cv.WND_PROP_AUTOSIZE) >= 0:
                    cv.imshow(window_title_left, frame_left)
                    cv.imshow(window_title_right, frame_right)
                else:
                    break 
                keyCode = cv.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    cv.imwrite("Left_img1.jpg", frame_left)
                    cv.imwrite("Right_img1.jpg", frame_right)
                    break
        finally:
            video_capture_left.release()
            video_capture_right.release()
            cv.destroyAllWindows()
    #image1 = cv.imread("../Data/Left_img1.jpg")
    #image2 = cv.imread("../Data/Right_img1.jpg")

    #image1 = cv.resize(image1, (int(image1.shape[1] * 0.5), int(image1.shape[0] * 0.5)), interpolation=cv.INTER_AREA)
    #image2 = cv.resize(image2, (int(image2.shape[1] * 0.5), int(image2.shape[0] * 0.5)), interpolation=cv.INTER_AREA)
    



if __name__ == "__main__":
    main()
