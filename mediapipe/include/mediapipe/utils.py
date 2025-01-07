import os
import cv2
import time
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

def load_camera_params(configFile, height):
    """
    Load camera parameters for the specified resolution.
    Args:
        configFile: Path to the .conf file
        height: Image height (pixels)
    Returns:
        Dictionary with intrinsic and extrinsic parameters for left and right cameras
    """
    # Check if Config File Exists and is Readable
    if not os.path.exists(configFile):
        print(f"Error: Config file '{configFile}' not found.")
        return None
    if not os.access(configFile, os.R_OK):
        print(f"Error: Config file '{configFile}' is not readable.")
        return None
    
    # Convert Height Pixels to Resolution
    resolution = 'VGA'
    if height == 1242:
        resolution = '2K'
    elif height == 1080:
        resolution = 'FHD'
    elif height == 720:
        resolution = 'HD'
    print(colored(f'\nZED Camera Resolution: {resolution}\n', 'green'))

    # Load Config File
    config = configparser.ConfigParser()
    config.read(configFile)

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


def run_mediapipe_2d(image, mpHands, mpDetector, mpDrawing):
    """
    Extract 3D Hand Keypoints using MediaPipe.
    Args:
        image: Input image for MediaPipe processing
        mpHands: Initialized Mediapipe Hands
        mpDetector: Initialized MediaPipe hands detector
        mpDrawing: Initialized Mediaped Drawing
    Returns:
        List of detected keypoints as Point messages
    """

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mpDetector.process(imageRGB)
    keypoints = []
    # Extract Mediapipe Hand Keypoints
    if results.multi_hand_landmarks:
        hand_side = results.multi_handedness[0].classification[0].label
        if hand_side == 'Right': # Right is the new left
            return None, image
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append(Point(x=landmark.x, 
                                       y=landmark.y, 
                                       z=landmark.z))
            # Draw Hand Keypoints
            mpDrawing.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

        return keypoints, image
    return None, image


def run_mediapipe_stereo(image, mpHands, mpDetector, mpDrawing):
    """
    Extract Hand Keypoints positions in pixels using MediaPipe.
    Args:
        image: Input image for MediaPipe processing
        mpHands: Initialized Mediapipe Hands
        mpDetector: Initialized MediaPipe hands detector
        mpDrawing: Initialized Mediaped Drawing
    Returns:
        List of detected keypoints as Point messages
    """

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = imageRGB.shape
    results = mpDetector.process(imageRGB)
    keypoints = []
    # Extract Mediapipe Hand Keypoints
    if results.multi_hand_landmarks:
        hand_side = results.multi_handedness[0].classification[0].label
        if hand_side == 'Right': # Right is the new left
            return None, image
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append(Point(x=landmark.x*width, 
                                       y=landmark.y*height, 
                                       z=0))
            # Draw Hand Keypoints
            mpDrawing.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

        return keypoints, image
    return None, image


def compute_3d_coordinates(leftKeypoints, rightKeypoints, cameraParams):
    """
    Compute 3D coordinates from stereo keypoints.
    Args:
        leftKeypoints: List of keypoints from the left image
        rightKeypoints: List of keypoints from the right image
        cameraParams: Dictionary containing 'baseline', 'left' and 'right' camera intrinsic parameters
    Returns:
        3D keypoints as a list of Point messages
    """

    baseline = cameraParams['stereo']['baseline'] / 1000.0
    fx = (cameraParams['left']['fx'] + cameraParams['right']['fx']) / 2.0
    fy = (cameraParams['left']['fy'] + cameraParams['right']['fy']) / 2.0
    cx = cameraParams['left']['cx']
    cy = cameraParams['left']['cy']
    
    keypoints_3d = []
    for left, right in zip(leftKeypoints, rightKeypoints):
        disparity = left.x - right.x
        if disparity <= 0:
            print(colored('ERROR: Invalid disparity calculation!', 'red'))
            return None
        Z = (fx * baseline) / disparity
        X = ((left.x - cx) * Z) / fx
        Y = ((left.y - cy) * Z) / fy
        keypoints_3d.append(Point(x=X, y=Y, z=Z))
    
    return keypoints_3d


def transform_keypoints(keypoints, wrist_if_mcp_dist=0.10):
    """
    Transform keypoints into a new frame where:
    - Wrist (keypoint 0) is at the origin (0, 0, 0)
    - Z-axis is defined by the vector from WRIST [KP0] to MIDDLE_FINGER_MCP [KP9]
    - X-axis is defined by the vector from INDEX_FINGER_MCP [KP5]  to MIDDLE_FINGER_MCP [KP9]
    Args:
        keypoints: The raw keypoints
        wrist_if_mcp_dist: Distance from Wrist to Index Finger Metacarpophalangeal
    Returns:
        transformed_keypoints: The transformed keypoints in the new coordinate frame
    """

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


def normalize(vector):
    """
    Normalizes a given vector to a unit vector.
    Args:
        vector: Input vector
    Returns:
        normalizedVector: Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    normalizedVector = vector / norm
    return normalizedVector


def map_keypoints_shadow(humanKeypoints):
    """
    Transform Hand Keypoints to match Shadow Hand dimensions.
    Args:
        keypoints: The raw keypoints
    Returns:
        shadow_keypoints: The new keypoints adapted to Shadow Hand
    """

    # Convert Keypoints to Arrays
    humanKeypointsArray = [np.array([kp.x, kp.y, kp.z]) for kp in humanKeypoints]

    # Convert Keypoints to Shadow Hand
    shadowKeypointsArray = [np.array([0.0, 0.0, 0.0]) for kp in humanKeypoints]

    # Thumb
    metacarpalSize = np.sqrt(29**2 + 34**2)
    shadowKeypointsArray[1] = normalize(humanKeypointsArray[1]-humanKeypointsArray[0]) * metacarpalSize / 1000
    shadowKeypointsArray[2] = shadowKeypointsArray[1] + normalize(humanKeypointsArray[2]-humanKeypointsArray[1]) * 38.0 / 1000
    shadowKeypointsArray[3] = shadowKeypointsArray[2] + normalize(humanKeypointsArray[3]-humanKeypointsArray[2]) * 32.0 / 1000
    shadowKeypointsArray[4] = shadowKeypointsArray[3] + normalize(humanKeypointsArray[4]-humanKeypointsArray[3]) * 27.5 / 1000
    # Forefinger
    metacarpalSize = np.sqrt(95**2 + 33**2)
    shadowKeypointsArray[5] = normalize(humanKeypointsArray[5]-humanKeypointsArray[0]) * metacarpalSize / 1000
    shadowKeypointsArray[6] = shadowKeypointsArray[5] + normalize(humanKeypointsArray[6]-humanKeypointsArray[5]) * 45.0 / 1000
    shadowKeypointsArray[7] = shadowKeypointsArray[6] + normalize(humanKeypointsArray[7]-humanKeypointsArray[6]) * 25.0 / 1000
    shadowKeypointsArray[8] = shadowKeypointsArray[7] + normalize(humanKeypointsArray[8]-humanKeypointsArray[7]) * 26.0 / 1000
    # Middlefinger
    metacarpalSize = np.sqrt(99**2 + 11**2)
    shadowKeypointsArray[9] = normalize(humanKeypointsArray[9]-humanKeypointsArray[0]) * metacarpalSize / 1000
    shadowKeypointsArray[10] = shadowKeypointsArray[9] + normalize(humanKeypointsArray[10]-humanKeypointsArray[9]) * 45.0 / 1000
    shadowKeypointsArray[11] = shadowKeypointsArray[10] + normalize(humanKeypointsArray[11]-humanKeypointsArray[10]) * 25.0 / 1000
    shadowKeypointsArray[12] = shadowKeypointsArray[11] + normalize(humanKeypointsArray[12]-humanKeypointsArray[11]) * 26.0 / 1000
    # Ringfinger
    metacarpalSize = np.sqrt(95**2 + 11**2)
    shadowKeypointsArray[13] = normalize(humanKeypointsArray[13]-humanKeypointsArray[0]) * metacarpalSize / 1000
    shadowKeypointsArray[14] = shadowKeypointsArray[13] + normalize(humanKeypointsArray[14]-humanKeypointsArray[13]) * 45.0 / 1000
    shadowKeypointsArray[15] = shadowKeypointsArray[14] + normalize(humanKeypointsArray[15]-humanKeypointsArray[14]) * 25.0 / 1000
    shadowKeypointsArray[16] = shadowKeypointsArray[15] + normalize(humanKeypointsArray[16]-humanKeypointsArray[15]) * 26.0 / 1000
    # Littlefinger
    metacarpalSize = np.sqrt(86.6**2 + 33**2)
    shadowKeypointsArray[17] = normalize(humanKeypointsArray[17]-humanKeypointsArray[0]) * metacarpalSize / 1000
    shadowKeypointsArray[18] = shadowKeypointsArray[17] + normalize(humanKeypointsArray[18]-humanKeypointsArray[17]) * 45.0 / 1000
    shadowKeypointsArray[19] = shadowKeypointsArray[18] + normalize(humanKeypointsArray[19]-humanKeypointsArray[18]) * 25.0 / 1000
    shadowKeypointsArray[20] = shadowKeypointsArray[19] + normalize(humanKeypointsArray[20]-humanKeypointsArray[19]) * 26.0 / 1000

    # Convert Keypoints back to Point
    shadowKeypoints = [Point(kp[0], kp[1], kp[2]) for kp in shadowKeypointsArray]

    return shadowKeypoints


def map_knuckles_shadow(handKeypoints, knucklesDist=0.022):

    # Convert Keypoints to Arrays
    handKeypointsArray = [np.array([kp.x, kp.y, kp.z]) for kp in handKeypoints]

    # Calculate Median Knuckles Distance
    knuckle_indices = [5, 9, 13, 17]
    knuckle_points = [handKeypointsArray[i] for i in knuckle_indices]
    distances = [np.linalg.norm(knuckle_points[i] - knuckle_points[i - 1])
                 for i in range(1, len(knuckle_points))]
    median_distance = np.median(distances)

    # Calculate the Scaling Factor
    scaling_factor = knucklesDist / median_distance

    # Apply scaling to the x and y coordinates (horizontal scaling)
    scaledKeypointsArray = [
        np.array([kp[0] * scaling_factor, kp[1] * scaling_factor, kp[2]])
        for kp in handKeypointsArray
    ]

    # Convert Keypoints back to Point
    scaledKeypoints = [Point(kp[0], kp[1], kp[2]) for kp in scaledKeypointsArray]

    return scaledKeypoints


def palm2wrist(keypoints):
    """
    Changes Hand Keypoints referential frame from 'rh_palm' to 'rh_wrist'.
    Args:
        keypoints: The Hand Keypoints in 'rh_palm' referential frame
    Returns:
        wristKeypoints: The Hand Keypoints in 'rh_wrist' referential frame
    """
    wristKeypoints = [Point(kp.x, kp.y, kp.z+(34.0/1000)) for kp in keypoints]
    return wristKeypoints
