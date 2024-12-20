#!/usr/bin/env python3

import rospy
from messages.msg import HandKeypoints 
from visualization_msgs.msg import Marker, MarkerArray

human = True
rvizPublisher = None

# Keypoint Connections
CONNECTIONS = [
    [1, 2, 3, 4],              # Thumb
    [5, 6, 7, 8],              # Index finger
    [9, 10, 11, 12],           # Middle finger
    [13, 14, 15, 16],          # Ring finger
    [17, 18, 19, 20],          # Pinky
    [0, 5, 9, 13, 17, 0, 1] # Palm
]

# Finger Colors (R, G, B, A)
FINGER_COLORS = {
    "thumb": (1.0, 0.0, 0.0, 1.0),  # Red
    "index": (1.0, 1.0, 0.0, 1.0),  # Yellow
    "middle": (0.0, 1.0, 0.0, 1.0), # Green
    "ring": (0.0, 0.0, 1.0, 1.0),   # Blue
    "pinky": (0.5, 0.0, 0.5, 1.0),  # Purple
    "palm": {
        "sphere": (1.0, 1.0, 1.0, 1.0), # White spheres
        "line": (0.0, 0.0, 0.0, 1.0)    # Black lines
    }
}

def create_marker(marker_id, ns, marker_type, points, color, scale):
    """
    Helper function to create a Marker message.
    """    
    marker = Marker()
    marker.header.frame_id = "rh_wrist"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    if marker_type == Marker.LINE_STRIP:
        marker.points = points
    else:
        marker.pose.position = points
    marker.pose.orientation.w = 1.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    return marker

def hand_keypoints_callback(msg):
    """
    Callback function to handle incoming hand keypoints messages.
    """

    global rvizPublisher, human

    marker_array = MarkerArray()

    # Human Case
    if human:
        # Set Markers Color
        sphereColor = (1.0, 0.0, 0.0, 1.0) # Red
        lineColor = (1.0, 1.0, 1.0, 1.0)   # White

        # Add Keypoints
        for i, keypoint in enumerate(msg.keypoints):
            sphere_marker = create_marker(
                marker_id=i,
                ns="keypoints",
                marker_type=Marker.SPHERE,
                points=keypoint,
                color=sphereColor,
                scale=(0.012, 0.012, 0.012) # Sphere size
            )
            marker_array.markers.append(sphere_marker)

        # Add Keypoints Connections
        line_id = len(msg.keypoints)
        for connection in CONNECTIONS:
            line_points = [msg.keypoints[idx] for idx in connection]
            line_marker = create_marker(
                marker_id=line_id,
                ns="connections",
                marker_type=Marker.LINE_STRIP,
                points=line_points,
                color=lineColor,
                scale=(0.005, 0.0, 0.0) # Line width
            )
            marker_array.markers.append(line_marker)
            line_id += 1

    # Shadow Remmaped Case
    else:
        line_id = len(msg.keypoints)
        # Add Keypoints and Connections
        for i, keypoint in enumerate(msg.keypoints):
            if i in range(1, 5):
                color = FINGER_COLORS["thumb"]
            elif i in range(5, 9):  
                color = FINGER_COLORS["index"]
            elif i in range(9, 13): 
                color = FINGER_COLORS["middle"]
            elif i in range(13, 17):
                color = FINGER_COLORS["ring"]
            elif i in range(17, 21): 
                color = FINGER_COLORS["pinky"]
            else:
                color = FINGER_COLORS["palm"]["sphere"]

            sphere_marker = create_marker(
                marker_id=i,
                ns="keypoints",
                marker_type=Marker.SPHERE,
                points=keypoint,
                color=color,
                scale=(0.012, 0.012, 0.012)  # Sphere size
            )
            marker_array.markers.append(sphere_marker)

        for connection in CONNECTIONS:
            line_points = [msg.keypoints[idx] for idx in connection]
            if connection[0] == 0:
                line_color = FINGER_COLORS["palm"]["line"]
            elif connection[0] == 1:
                line_color = FINGER_COLORS["thumb"]
            elif connection[0] == 5: 
                line_color = FINGER_COLORS["index"]
            elif connection[0] == 9:
                line_color = FINGER_COLORS["middle"]
            elif connection[0] == 13:
                line_color = FINGER_COLORS["ring"]
            elif connection[0] == 17:
                line_color = FINGER_COLORS["pinky"]
            else:
                line_color = FINGER_COLORS["palm"]["line"]

            line_marker = create_marker(
                marker_id=line_id,
                ns="connections",
                marker_type=Marker.LINE_STRIP,
                points=line_points,
                color=line_color,
                scale=(0.005, 0.0, 0.0)  # Line width
            )
            marker_array.markers.append(line_marker)
            line_id += 1

    # Publish markers
    rvizPublisher.publish(marker_array)


def main():
    
    global rvizPublisher, human

    # Init ROS
    rospy.init_node("hand_keypoints_visualizer")

    # Get ROS Parameters
    human = rospy.get_param('~human', 'True')

    if human:
        # Subscriber to Human Hand Keypoints
        rospy.Subscriber('/human_hand_keypoints', HandKeypoints, hand_keypoints_callback)
        # Publisher for Visualization Markers
        rvizPublisher = rospy.Publisher('/human_hand_keypoints_rviz', MarkerArray, queue_size=1)
    else:
        # Subscriber to Shadow Hand Keypoints
        rospy.Subscriber('/shadow_hand_keypoints', HandKeypoints, hand_keypoints_callback)
        # Publisher for Visualization Markers
        rvizPublisher = rospy.Publisher('/shadow_hand_keypoints_rviz', MarkerArray, queue_size=1)

    rospy.spin()



if __name__ == "__main__":
    main()