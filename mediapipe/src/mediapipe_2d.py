#!/usr/bin/env python3

import cv2
import rospy
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from messages.msg import HandKeypoints
from std_msgs.msg import Header

mp_hands = None
detect_hands = None
mp_drawing = None
keypointsPublisher = None

def image_callback(msg):
    global mp_hands, detect_hands, mp_drawing, keypointsPublisher

    # Convert image to OpenCV
    bridge = CvBridge()
    cvImage = bridge.imgmsg_to_cv2(msg, "bgr8")

    # Mediapipe
    imgRGB = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    result = detect_hands.process(imgRGB)

    # Prepare the custom message
    handKeypointsMsg = HandKeypoints()
    handKeypointsMsg.header = Header()
    handKeypointsMsg.header.stamp = rospy.Time.now()

    # Extract and Display Mediapipe Hand Keypoints
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                point = Point()
                point.x = landmark.x
                point.y = landmark.y
                point.z = landmark.z
                keypoints.append(point)
            handKeypointsMsg.keypoints = keypoints

            # Publish Hand Keypoints
            keypointsPublisher.publish(handKeypointsMsg)

            # Draw Hand Keypoints
            mp_drawing.draw_landmarks(cvImage, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the image using OpenCV
    cv2.imshow("ZED Left Image", cvImage)
    
    # Wait for 'q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Key 'q' pressed, shutting down")
        return

def main():
    global mp_hands, detect_hands, mp_drawing, keypointsPublisher

    # Init ROS
    rospy.init_node('zed_left_image_subscriber', anonymous=True)

    # Get ROS Parameters
    image_topic = rospy.get_param('~image_topic', '/zed/left_image')
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
