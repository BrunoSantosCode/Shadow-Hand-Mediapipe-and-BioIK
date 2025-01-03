#!/usr/bin/env python3

#* * * * * * * get_ur5_joints.py * * * * * * *#
#*  Displays UR5 current joint angle values  *#
#* * * * * * * * * * * * * * * * * * * * * * *#

import rospy
import numpy as np
from sr_robot_commander.sr_arm_commander import SrArmCommander

if __name__ == "__main__":
    # Init ROS
    rospy.init_node('get_ur5_joints')

    # Shadow Hand commander
    arm_commander = SrArmCommander(name='right_arm')

    # Get joint positions
    joints_position = arm_commander.get_joints_position()

    # Filter UR5 joints
    ur5_joints = {k: v for k, v in joints_position.items() if k.startswith('ra_')}

    # Display UR5 joint values in radians and degrees
    print("\nUR5 Joint Positions:")
    for joint, radians in ur5_joints.items():
        degrees = np.degrees(radians)
        print(f"{joint}: {radians:.4} rad ({degrees:.2f} degrees)")