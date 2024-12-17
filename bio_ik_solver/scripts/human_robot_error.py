#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import Point
from messages.msg import HandKeypoints

DISPLAY_NAMES = {
    "rh_thtip": "Thumb Tip",
    "rh_fftip": "Forefinger Tip",
    "rh_mftip": "Middle Finger Tip",
    "rh_rftip": "Ring Finger Tip",
    "rh_lftip": "Little Finger Tip",
    "rh_thdistal": "Thumb Distal",
    "rh_ffdistal": "Forefinger Distal",
    "rh_mfdistal": "Middle Finger Distal",
    "rh_rfdistal": "Ring Finger Distal",
    "rh_lfdistal": "Little Finger Distal",
    "rh_thmiddle": "Thumb Middle",
    "rh_ffmiddle": "Forefinger Middle",
    "rh_mfmiddle": "Middle Finger Middle",
    "rh_rfmiddle": "Ring Finger Middle",
    "rh_lfmiddle": "Little Finger Middle",
}

MEDIAPIPE_KEYPOINTS = {
    "rh_thtip": 4, "rh_fftip": 8, "rh_mftip": 12, "rh_rftip": 16, "rh_lftip": 20,
    "rh_thdistal": 3, "rh_ffdistal": 7, "rh_mfdistal": 11, "rh_rfdistal": 15, "rh_lfdistal": 19,
    "rh_thmiddle": 2, "rh_ffmiddle": 6, "rh_mfmiddle": 10, "rh_rfmiddle": 14, "rh_lfmiddle": 18,
}

def euclideanDist(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

def getRobotPos(tf_listener, frame_name, reference_frame="rh_wrist"):
    """
    Get the position of a robotic hand frame relative to a reference frame.
    Default reference frame is 'rh_wrist'.
    """
    try:
        tf_listener.waitForTransform(reference_frame, frame_name, rospy.Time(0), rospy.Duration(1.0))
        (trans, _) = tf_listener.lookupTransform(reference_frame, frame_name, rospy.Time(0))
        return Point(x=trans[0], y=trans[1], z=trans[2])
    except (tf.Exception, tf.LookupException, tf.ConnectivityException):
        rospy.logerr(f"Failed to get transform for frame: {frame_name}")
        return None

def keypoints_callback(msg, tf_listener):
    """Callback for human hand keypoints message."""
    human_positions = msg.keypoints
    distances = {}

    for robot_frame, human_index in MEDIAPIPE_KEYPOINTS.items():
        if human_index >= len(human_positions):
            rospy.logerr(f"Invalid index {human_index} for keypoints array.")
            continue

        # Get human keypoint position
        human_position = human_positions[human_index]

        # Get robotic keypoint position
        robot_position = getRobotPos(tf_listener, robot_frame)
        if robot_position is None:
            continue

        # Calculate distance
        distance = euclideanDist(human_position, robot_position) * 100  # Convert to centimeters
        distances[DISPLAY_NAMES[robot_frame]] = distance

    # Print or log the distances
    print("\033[2J\033[H")  # Clear terminal
    print("{:<25} {:>10}".format("Keypoint", "Distance (cm)"))
    print("-" * 37)
    for friendly_name, distance in distances.items():
        print("{:<25} {:>10.2f}".format(friendly_name, distance))

def main():
    # Init ROS
    rospy.init_node("human_robot_error")
    tf_listener = tf.TransformListener()

    # Get ROS Parameters
    keypoints_topic = rospy.get_param('~keypoints_topic', '/shadow_hand_keypoints')

    # Create subscriber
    rospy.Subscriber(keypoints_topic, HandKeypoints, keypoints_callback, callback_args=tf_listener)

    rospy.loginfo("'human_robot_error' node started.")

    rospy.spin()

if __name__ == "__main__":
    main()
