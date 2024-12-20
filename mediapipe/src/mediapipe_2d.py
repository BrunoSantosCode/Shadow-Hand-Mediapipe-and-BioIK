#!/usr/bin/env python3

import sys
import rospkg
sys.path.append(rospkg.RosPack().get_path('mediapipe'))

from include.mediapipe.utils import *


mpHands = None
mpDrawing = None
mpDetectHand = None
wrist_if_mcp_dist = None
humanKeypointsPublisher = None
shadowKeypointsPublisher = None
lastTime = None

def image_callback(msg):
    global mpHands, mpDetectHand, mpDrawing, wrist_if_mcp_dist
    global humanKeypointsPublisher, shadowKeypointsPublisher
    global lastTime

    # Convert image to OpenCV
    bridge = CvBridge()
    cvImage = bridge.imgmsg_to_cv2(msg, "bgr8")

    # Mediapipe
    keypoints, cvImage = run_mediapipe_2d(cvImage, mpHands, mpDetectHand, mpDrawing)

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
        humanKeypoints = transform_keypoints(keypoints, wrist_if_mcp_dist) 
        wristHumanKeypoints = palm2wrist(humanKeypoints) 
        
        # Map Human Hand to Shadow Hand
        shadowKeypoints = map_keypoints_shadow(humanKeypoints)
        shadowKeypoints = map_knuckles_shadow(shadowKeypoints)
        wristShadowKeypoints = palm2wrist(shadowKeypoints)

        # Publish Hand Keypoints
        humanKeypointsMsg.keypoints = wristHumanKeypoints
        humanKeypointsPublisher.publish(humanKeypointsMsg)
        shadowKeypointsMsg.keypoints = wristShadowKeypoints
        shadowKeypointsPublisher.publish(shadowKeypointsMsg)

    # Resize image to display
    origHeight, origWidth = cvImage.shape[:2]
    newWidth = int( 720 * (origWidth/origHeight) )
    displayImage = cv2.resize(cvImage, (newWidth, 720), interpolation=cv2.INTER_AREA)
    
    # Display FPS
    currentTime = time.perf_counter()
    fps = 1/(currentTime-lastTime)
    lastTime = currentTime
    cv2.putText(displayImage, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # Display the image using OpenCV
    cv2.imshow("Image", displayImage)
    
    # Wait for 'q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Key 'q' pressed, shutting down")
        return

def main():
    global mpHands, mpDetectHand, mpDrawing
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
    mpDetectHand = mpHands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, min_detection_confidence=0.75)
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