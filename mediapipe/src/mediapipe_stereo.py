#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from messages.msg import HandKeypoints
from std_msgs.msg import Header

mp_hands = None
mp_drawing = None
detect_hands = None
wrist_if_mcp_dist = None
keypointsPublisher = None

def transform_keypoints(keypoints):
    """
    Transform keypoints into a new frame where:
    - Wrist (keypoint 0) is at the origin (0, 0, 0)
    - Z-axis is defined by the vector from WRIST [KP0] to MIDDLE_FINGER_MCP [KP9]
    - X-axis is defined by the vector from INDEX_FINGER_MCP [KP5]  to MIDDLE_FINGER_MCP [KP9]
    Args:
        keypoints: The raw keypoints.
    Returns:
        transformed_keypoints: The transformed keypoints in the new coordinate frame.
    """

    global wrist_if_mcp_dist

    wrist = keypoints[0]

    # Set Origin in WRIST
    translated_keypoints = []
    for keypoint in keypoints:
        translated_keypoints.append(Point(x=keypoint.x - wrist.x, y=keypoint.y - wrist.y, z=keypoint.z - wrist.z))

    # Get Keypoints for Frame Reference
    wrist = translated_keypoints[0]
    ringFingerMcp = translated_keypoints[13]
    middleFingerMcp = translated_keypoints[9]

    # Set New Coordinate Frame 
    wrist_mf_mcp = np.array([middleFingerMcp.x - wrist.x, middleFingerMcp.y - wrist.y, middleFingerMcp.z - wrist.z])
    wrist_rf_mcp = np.array([ringFingerMcp.x - wrist.x, ringFingerMcp.y - wrist.y, ringFingerMcp.z - wrist.z])
    z_dir = (wrist_mf_mcp+wrist_rf_mcp)/2.0
    if np.linalg.norm(z_dir) != 0:
        z_dir = z_dir/np.linalg.norm(z_dir)
    y_dir = np.cross(wrist_rf_mcp, wrist_mf_mcp)
    if np.linalg.norm(y_dir) != 0:
        y_dir = y_dir/np.linalg.norm(y_dir)
    x_dir = np.cross(y_dir, z_dir)
    if np.linalg.norm(x_dir) != 0:
        x_dir = x_dir/np.linalg.norm(x_dir)

    # Set scaling factor
    indexFingerMcp = translated_keypoints[5]
    wrist_if_mcp = np.array([indexFingerMcp.x, indexFingerMcp.y, indexFingerMcp.z])
    scaleFactor = wrist_if_mcp_dist / np.linalg.norm(wrist_if_mcp)

    # Apply New Reference Frame to Keypoints
    rotation_matrix = np.column_stack([x_dir, y_dir, z_dir]).T
    transformed_keypoints = []
    for keypoint in translated_keypoints:
        point_vec = np.array([keypoint.x, keypoint.y, keypoint.z])
        transformed_point = np.dot(rotation_matrix, point_vec)
        scaled_point = transformed_point * scaleFactor
        transformed_keypoints.append(Point(x=scaled_point[0], y=scaled_point[1], z=scaled_point[2]))
    
    return transformed_keypoints

def image_callback(msg):
    global mp_hands, detect_hands, mp_drawing, keypointsPublisher

    # Convert image to OpenCV
    bridge = CvBridge()
    cvImage = bridge.imgmsg_to_cv2(msg, "bgr8")

    # Split Stereo Image into Left and Right Images
    _, width, _ = cvImage.shape
    singleImageWidth = width // 2
    cvLeftImage = cvImage[:, :singleImageWidth]
    cvRightImage = cvImage[:, singleImageWidth:]

    # # Mediapipe
    # imgRGB = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    # result = detect_hands.process(imgRGB)

    # # Prepare the custom message
    # handKeypointsMsg = HandKeypoints()
    # handKeypointsMsg.header = Header()
    # handKeypointsMsg.header.stamp = rospy.Time.now()

    # # Extract and Display Mediapipe Hand Keypoints
    # if result.multi_hand_landmarks:
    #     for hand_landmarks in result.multi_hand_landmarks:
    #         keypoints = []
    #         for landmark in hand_landmarks.landmark:
    #             point = Point()
    #             point.x = landmark.x
    #             point.y = landmark.y
    #             point.z = landmark.z
    #             keypoints.append(point)
            
    #         # Reorient Keypoints
    #         newKeypoints = transform_keypoints(keypoints) 
    #         handKeypointsMsg.keypoints = newKeypoints

    #         # Publish Hand Keypoints
    #         keypointsPublisher.publish(handKeypointsMsg)

    #         # Draw Hand Keypoints
    #         mp_drawing.draw_landmarks(cvImage, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the image using OpenCV
    cv2.imshow("Left Image", cvLeftImage)
    cv2.imshow("Right Image", cvRightImage)
    
    # Wait for 'q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Key 'q' pressed, shutting down")
        return

def main():
    global mp_hands, detect_hands, mp_drawing, keypointsPublisher, wrist_if_mcp_dist

    # Init ROS
    rospy.init_node('zed_left_image_subscriber', anonymous=True)

    # Get ROS Parameters
    wrist_if_mcp_dist = rospy.get_param('~wrist_if_mcp_topic', '0.10')
    image_topic = rospy.get_param('~image_topic', '/zed/stereo_image')
    keypoints_topic = rospy.get_param('~keypoints_topic', '/hand_keypoints')

    # Init Mediapipe
    mp_hands = mp.solutions.hands
    detect_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils
    
    # Create ROS Subscriber
    rospy.Subscriber(image_topic, Image, image_callback)

    # Create ROS Publisher
    keypointsPublisher = rospy.Publisher(keypoints_topic, HandKeypoints, queue_size=10)

    # Spin
    rospy.spin()

if __name__ == '__main__':
    main()
