#!/usr/bin/env python3

import os
import rospy
import rospkg
import rosbag
from sensor_msgs.msg import Image

def record_single_message():
    rospy.init_node('rosbag_record', anonymous=True)

    # Get the path to the package dynamically using rospkg
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('image')
    
    # Define the rosbags directory within the package
    bag_dir = os.path.join(package_path, 'rosbags')

    # Get bag file name from parameter
    bag_name = rospy.get_param('~rosbag_name', 'default')
    bag_path = os.path.join(bag_dir, f"{bag_name}.bag")

    # Create the rosbags directory if it doesn't exist
    if not os.path.exists(bag_dir):
        os.makedirs(bag_dir)

    rospy.loginfo(f"Recording messages to {bag_path}")
    bag = rosbag.Bag(bag_path, 'w')

    try:
        # Wait for a single message from each topic
        left_image_msg = rospy.wait_for_message('/zed/left_image', Image)
        stereo_image_msg = rospy.wait_for_message('/zed/stereo_image', Image)

        # Write the messages to the bag
        bag.write('/zed/left_image', left_image_msg)
        bag.write('/zed/stereo_image', stereo_image_msg)

        rospy.loginfo("Recorded one message from each topic.")
    except rospy.ROSException as e:
        rospy.logerr(f"Failed to record messages: {e}")
    finally:
        bag.close()

if __name__ == "__main__":
    record_single_message()
