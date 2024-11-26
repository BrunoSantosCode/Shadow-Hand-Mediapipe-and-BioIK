#!/usr/bin/env python3

import os
import cv2
import rospy
import rospkg
import numpy as np
import configparser
import mediapipe as mp
from termcolor import colored
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from messages.msg import HandKeypoints
from std_msgs.msg import Header

mp_hands = None
mp_drawing = None
detect_hands = None
camera_params = None
camera_resolution = None
wrist_if_mcp_dist = None
keypointsPublisher = None

def load_camera_params(config_file, resolution):
    """
    Load camera parameters for the specified resolution
    Args:
        config_file: Path to the .conf file
        resolution: Camera resolution as a string ('VGA', 'HD', 'FHD' or '2K')
    Returns:
        Dictionary with intrinsic and extrinsic parameters for left and right cameras
    """
    # Check if Config File Exists and is Readable
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        return None
    if not os.access(config_file, os.R_OK):
        print(f"Error: Config file '{config_file}' is not readable.")
        return None

    # Load Config File
    config = configparser.ConfigParser()
    config.read(config_file)

    # Build and Check Resolution-specific Keys
    left_key = f"LEFT_CAM_{resolution}"
    right_key = f"RIGHT_CAM_{resolution}"
    if not config.has_section(left_key):
        raise KeyError(f"Section '{left_key}' not found in config file.")
    if not config.has_section(right_key):
        raise KeyError(f"Section '{right_key}' not found in config file.")
    
    # Extract Camera Parameters for Specified Resolution
    params = {
        'left': {
            'fx': float(config[left_key]['fx']),
            'fy': float(config[left_key]['fy']),
            'cx': float(config[left_key]['cx']),
            'cy': float(config[left_key]['cy']),
            'k1': float(config[left_key]['k1']),
            'k2': float(config[left_key]['k2']),
            'p1': float(config[left_key]['p1']),
            'p2': float(config[left_key]['p2']),
            'k3': float(config[left_key]['k3'])
        },
        'right': {
            'fx': float(config[right_key]['fx']),
            'fy': float(config[right_key]['fy']),
            'cx': float(config[right_key]['cx']),
            'cy': float(config[right_key]['cy']),
            'k1': float(config[right_key]['k1']),
            'k2': float(config[right_key]['k2']),
            'p1': float(config[right_key]['p1']),
            'p2': float(config[right_key]['p2']),
            'k3': float(config[right_key]['k3'])
        },
        'stereo': {
            'baseline': float(config['STEREO']['Baseline']),
            'ty': float(config['STEREO']['TY']),
            'tz': float(config['STEREO']['TZ'])
        }
    }
    return params

def run_mediapipe(image, detector):
    """
    Extract hand keypoints using MediaPipe.
    Args:
        image: Input image for MediaPipe processing
        detector: Initialized MediaPipe hands detector
    Returns:
        List of detected keypoints as Point messages
    """
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = imageRGB.shape
    results = detector.process(imageRGB)
    keypoints = []
    # Extract Mediapipe Hand Keypoints
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append(Point(x=landmark.x*width, 
                                       y=landmark.y*height, 
                                       z=0))
            # Draw Hand Keypoints
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return keypoints, image
    return None, image

def compute_3d_coordinates(left_keypoints, right_keypoints, camera_params):
    """
    Compute 3D coordinates from stereo keypoints
    Args:
        left_keypoints: List of keypoints from the left image
        right_keypoints: List of keypoints from the right image
        camera_params: Dictionary containing 'baseline', 'left' and 'right' camera intrinsic parameters
    Returns:
        3D keypoints as a list of Point messages
    """

    baseline = camera_params['stereo']['baseline']
    fx = camera_params['left']['fx']
    fy = camera_params['left']['fy']
    cx = camera_params['left']['cx']
    cy = camera_params['left']['cy']
    
    keypoints_3d = []
    for left, right in zip(left_keypoints, right_keypoints):
        disparity = left.x - right.x
        if disparity <= 0:
            print(colored('ERROR: Invalid disparity calculation!', 'red'))
            return None
        Z = (fx * baseline) / disparity
        X = ((left.x - cx) * Z) / fx
        Y = ((left.y - cy) * Z) / fy
        keypoints_3d.append(Point(x=X, y=Y, z=Z))
    
    return keypoints_3d

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

    print(Keypoints)

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
    global mp_hands, detect_hands, mp_drawing, camera_params, keypointsPublisher

    # Convert image to OpenCV
    bridge = CvBridge()
    cvImage = bridge.imgmsg_to_cv2(msg, "bgr8")

    # Split Stereo Image into Left and Right Images
    _, width, _ = cvImage.shape
    singleImageWidth = width // 2
    cvLeftImage = cvImage[:, :singleImageWidth]
    cvRightImage = cvImage[:, singleImageWidth:]

    # Mediapipe
    leftKeypoints, cvLeftImage = run_mediapipe(cvLeftImage, detect_hands)
    rightKeypoints, cvRightImage = run_mediapipe(cvRightImage, detect_hands)

    # If Hand Keypoints Detected
    if leftKeypoints and rightKeypoints and len(leftKeypoints) == len(rightKeypoints):
        
        # Compute 3D Keypoints
        keypoints = compute_3d_coordinates(leftKeypoints, rightKeypoints, camera_params)

        # Check 3D Keypoints
        if keypoints:

            # Prepare the custom message
            handKeypointsMsg = HandKeypoints()
            handKeypointsMsg.header = Header()
            handKeypointsMsg.header.stamp = rospy.Time.now()
                    
            # Reorient Keypoints
            newKeypoints = transform_keypoints(keypoints) 
            handKeypointsMsg.keypoints = newKeypoints

            # Publish Hand Keypoints
            keypointsPublisher.publish(handKeypointsMsg)

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
    global mp_hands, detect_hands, mp_drawing
    global keypointsPublisher, camera_resolution, wrist_if_mcp_dist
    global camera_params

    # Init ROS
    rospy.init_node('zed_left_image_subscriber', anonymous=True)

    # Get ROS Parameters
    camera_resolution = rospy.get_param('~camera_resolution', 'HD')
    wrist_if_mcp_dist = rospy.get_param('~wrist_if_mcp_topic', '0.10')
    image_topic = rospy.get_param('~image_topic', '/zed/stereo_image')
    keypoints_topic = rospy.get_param('~keypoints_topic', '/hand_keypoints')

    # Load Camera Parameters
    package_path = rospkg.RosPack().get_path('mediapipe')
    config_file = package_path + '/conf/camera.conf'
    camera_params = load_camera_params(config_file, camera_resolution)

    # Init Mediapipe
    mp_hands = mp.solutions.hands
    detect_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils
    
    # Create ROS Subscriber
    rospy.Subscriber(image_topic, Image, image_callback)

    # Create ROS Publisher
    keypointsPublisher = rospy.Publisher(keypoints_topic, HandKeypoints, queue_size=1)

    # Spin
    rospy.spin()

if __name__ == '__main__':
    main()
