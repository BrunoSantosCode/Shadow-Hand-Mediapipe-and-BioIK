#!/usr/bin/env python3

import cv2
import time
import rospy
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from messages.msg import HandKeypoints
from std_msgs.msg import Header

mpHands = None
mpDrawing = None
detectHand = None
wrist_if_mcp_dist = None
humanKeypointsPublisher = None
shadowKeypointsPublisher = None
lastTime = None

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
            mpDrawing.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

        return keypoints, image
    return None, image

def transform_keypoints(keypoints):
    """
    Transform keypoints into a new frame where:
    - Wrist (keypoint 0) is at the origin (0, 0, 0)
    - Z-axis is defined by the vector from WRIST [KP0] to MIDDLE_FINGER_MCP [KP9]
    - X-axis is defined by the vector from INDEX_FINGER_MCP [KP5]  to MIDDLE_FINGER_MCP [KP9]
    Args:
        keypoints: The raw keypoints
    Returns:
        transformed_keypoints: The transformed keypoints in the new coordinate frame
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

def normalize(vector):
    """
    Normalizes a given vector to a unit vector
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
    Transform keypoints to match Shadow Hand dimensions:
    Args:
        keypoints: The raw keypoints
    Returns:
        shadow_keypoints: The new keypoints adapted to Shadow Hand
    """

    # Convert Keypoints to Arrays
    humanKeypointsArray = [np.array([kp.x, kp.y, kp.z]) for kp in humanKeypoints]

    # Convert Keypoints to Shadow Hand
    shadowKeypointsArray = [np.array([0.0, 0.0, 0.0]) for kp in humanKeypoints]
    shadowKeypointsArray[0] = humanKeypointsArray[0]

    # Thumb
    metacarpalSize = np.sqrt(29**2 + 34**2)
    shadowKeypointsArray[1] = normalize(humanKeypointsArray[1]-humanKeypointsArray[0]) * metacarpalSize / 1000
    shadowKeypointsArray[2] = shadowKeypointsArray[1] + normalize(humanKeypointsArray[2]-humanKeypointsArray[1]) * 38.0 / 1000
    shadowKeypointsArray[3] = shadowKeypointsArray[2] + normalize(humanKeypointsArray[3]-humanKeypointsArray[2]) * 32.0 / 1000
    shadowKeypointsArray[4] = shadowKeypointsArray[3] + normalize(humanKeypointsArray[4]-humanKeypointsArray[3]) * 37.5 / 1000
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

def palm2wrist(keypoints):
    """
    Changes keypoints referential frame from 'rh_palm' to 'rh_wrist'
    Args:
        keypoints: The raw keypoints
    Returns:
        wristKeypoints: The keypoints in 'rh_wrist' referential frame
    """
    wristKeypoints = [Point(kp.x, kp.y, kp.z+(34.0/1000)) for kp in keypoints]
    return wristKeypoints

def image_callback(msg):
    global mpHands, detectHand, mpDrawing
    global humanKeypointsPublisher, shadowKeypointsPublisher
    global lastTime

    # Convert image to OpenCV
    bridge = CvBridge()
    cvImage = bridge.imgmsg_to_cv2(msg, "bgr8")

    # Mediapipe
    keypoints, cvImage = run_mediapipe(cvImage, detectHand)

    # If Hand Keypoints Detected
    if keypoints:
            
        # Prepare the custom message
        humanKeypointsMsg = HandKeypoints()
        humanKeypointsMsg.header = Header()
        humanKeypointsMsg.header.stamp = rospy.Time.now()
        shadowKeypointsMsg = HandKeypoints()
        shadowKeypointsMsg.header = Header()
        shadowKeypointsMsg.header.stamp = rospy.Time.now()

        # Reorient Keypoints
        humanKeypoints = transform_keypoints(keypoints) 
        wristHumanKeypoints = palm2wrist(humanKeypoints) 
        
        # Map Human Hand to Shadow Hand
        shadowKeypoints = map_keypoints_shadow(humanKeypoints)
        wristShadowKeypoints = palm2wrist(shadowKeypoints)

        # Publish Hand Keypoints
        humanKeypointsMsg.keypoints = wristHumanKeypoints
        humanKeypointsPublisher.publish(humanKeypointsMsg)
        shadowKeypointsMsg.keypoints = wristShadowKeypoints
        shadowKeypointsPublisher.publish(shadowKeypointsMsg)
    
    # Display FPS
    currentTime = time.perf_counter()
    fps = 1/(currentTime-lastTime)
    lastTime = currentTime
    cv2.putText(cvImage, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # Display the image using OpenCV
    cv2.imshow("Image", cvImage)
    
    # Wait for 'q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Key 'q' pressed, shutting down")
        return

def main():
    global mpHands, detectHand, mpDrawing
    global humanKeypointsPublisher, shadowKeypointsPublisher, wrist_if_mcp_dist
    global lastTime

    # Init ROS
    rospy.init_node('zed_left_image_subscriber', anonymous=True)

    # Get ROS Parameters
    wrist_if_mcp_dist = rospy.get_param('~wrist_if_mcp_topic', '0.10')
    image_topic = rospy.get_param('~image_topic', '/zed/left_image')
    human_keypoints_topic = rospy.get_param('~keypoints_topic', '/human_hand_keypoints')
    shadow_keypoints_topic = rospy.get_param('~keypoints_topic', '/shadow_hand_keypoints')

    # Init Mediapipe
    mpHands = mp.solutions.hands
    detectHand = mpHands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.75)
    mpDrawing = mp.solutions.drawing_utils
    
    # Create ROS Subscriber
    rospy.Subscriber(image_topic, Image, image_callback)
    lastTime = time.perf_counter()

    # Create ROS Publisher
    humanKeypointsPublisher = rospy.Publisher(human_keypoints_topic, HandKeypoints, queue_size=1)
    shadowKeypointsPublisher = rospy.Publisher(shadow_keypoints_topic, HandKeypoints, queue_size=1)

    # Spin
    rospy.spin()

if __name__ == '__main__':
    main()
