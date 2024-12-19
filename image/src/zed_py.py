#!/usr/bin/env python3

import cv2
import rospy
import pyzed.sl as sl
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def zed():
    # Init ROS
    rospy.init_node('zed_py')

    # Get ROS Parameters
    resolution_param = rospy.get_param('~camera_resolution', 'HD')
    depth_mode_param = rospy.get_param('~depth_mode', 'PERFORMANCE')
    left_topic = rospy.get_param('~left_image_topic', '/zed/left_image')
    #right_topic = rospy.get_param('~right_image_topic', '/zed/right_image')
    stereo_topic = rospy.get_param('~stereo_image_topic', '/zed/stereo_image')

    # Map resolution parameter to ZED SDK resolution
    if resolution_param == 'VGA':
        camera_resolution = sl.RESOLUTION.VGA
    elif resolution_param == 'HD':
        camera_resolution = sl.RESOLUTION.HD720
    elif resolution_param == 'FHD':
        camera_resolution = sl.RESOLUTION.HD1080
    else:
        camera_resolution = sl.RESOLUTION.HD2K

    # Map depth mode parameter to ZED SDK depth mode
    if depth_mode_param == 'PERFORMANCE':
        depth_mode = sl.DEPTH_MODE.PERFORMANCE
    elif depth_mode_param == 'QUALITY':
        depth_mode = sl.DEPTH_MODE.QUALITY
    elif depth_mode_param == 'ULTRA':
        depth_mode = sl.DEPTH_MODE.ULTRA
    else:
        depth_mode = sl.DEPTH_MODE.NEURAL

    # Create ROS Publishers
    leftImagePub = rospy.Publisher(left_topic, Image, queue_size=1)
    #rightImagePub = rospy.Publisher(right_topic, Image, queue_size=1)
    stereoImagePub = rospy.Publisher(stereo_topic, Image, queue_size=1)

    # Create a CvBridge object to convert OpenCV images to ROS Image messages
    bridge = CvBridge()

    # Create a Camera object
    zed = sl.Camera()

    # Create configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = camera_resolution
    init_params.depth_mode = depth_mode

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()

    # ZED image variables
    leftImage = sl.Mat()
    rightImage = sl.Mat()

    while not rospy.is_shutdown():

        # Grab an image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            # Capture ZED data
            zed.retrieve_image(leftImage, sl.VIEW.LEFT)
            zed.retrieve_image(rightImage, sl.VIEW.RIGHT)

            # Convert to OpenCV
            cvLeftImage = cv2.cvtColor(leftImage.get_data(), cv2.COLOR_BGRA2BGR)
            cvRightImage = cv2.cvtColor(rightImage.get_data(), cv2.COLOR_BGRA2BGR)
            cvStereoImage = cv2.hconcat([cvLeftImage, cvRightImage])

            # Convert to ROS
            rosLeftImage = bridge.cv2_to_imgmsg(cvLeftImage, "bgr8")
            #rosRightImage = bridge.cv2_to_imgmsg(cvRightImage, "bgr8")
            rosStereoImage = bridge.cv2_to_imgmsg(cvStereoImage, "bgr8")

            # Publish to ROS
            leftImagePub.publish(rosLeftImage)
            #rightImagePub.publish(rosRightImage)
            stereoImagePub.publish(rosStereoImage)
    
    # Close ZED camera
    zed.close()

if __name__ == '__main__':
    try:
        zed()
    except rospy.ROSInterruptException:
        pass
