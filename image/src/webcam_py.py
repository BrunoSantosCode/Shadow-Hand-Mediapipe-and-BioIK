#!/usr/bin/env python3

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def webcam():
    # Init ROS
    rospy.init_node('webcam_py')

    # Get ROS Parameters
    resolution_param = rospy.get_param('~camera_resolution', 'SD')
    image_topic = rospy.get_param('~image_topic', '/zed/left_image')

    # Create ROS Publisher
    imagePub = rospy.Publisher(image_topic, Image, queue_size=1)

    # Create a CvBridge object to convert OpenCV images to ROS Image messages
    bridge = CvBridge()

    # Open Webcam
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Couldn't open webcam.")
        exit()

    # Set Webcam resolution
    if resolution_param == 'HD1080':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    elif resolution_param == 'HD720':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not rospy.is_shutdown():

        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't grab frame from webcam.")
            break

        # Convert to ROS
        rosImage = bridge.cv2_to_imgmsg(frame, "bgr8")

        # Publish to ROS
        imagePub.publish(rosImage)
    
    # Close Webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        webcam()
    except rospy.ROSInterruptException:
        pass
