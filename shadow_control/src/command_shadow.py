#!/usr/bin/env python3

#* * * * * * * command_shadow.py * * * * * * *#
#*  INPUT: Shadow Hand joint angles          *#
#*  OUTPUT: Shadow Hand control commands     *#
#* * * * * * * * * * * * * * * * * * * * * * *#

import rospy
import numpy as np
from termcolor import colored
from std_msgs.msg import Float64MultiArray
from sr_robot_commander.sr_hand_commander import SrHandCommander

# Joints order
# rh_WRJ2  rh_WRJ1
# rh_FFJ4  rh_FFJ3  rh_FFJ2  rh_FFJ1 
# rh_LFJ5  rh_LFJ4  rh_LFJ3  rh_LFJ1
# rh_MFJ4  rh_MFJ3  rh_MFJ2  rh_MFJ1
# rh_RFJ4  rh_RFJ3  rh_RFJ2  rh_RFJ1
# rh_THJ5  rh_THJ4  rh_THJ3  rh_THJ2  rh_THJ1

# Define joints bounds
joints_min = [-28.0, -40.0,
              -20.0, -15.0,   0.0,  0.0,
                0.0, -20.0, -15.0,  0.0, 0.0,
              -20.0, -15.0,   0.0,  0.0,
              -20.0, -15.0,   0.0,  0.0,
              -60.0,   0.0, -12.0, -30.0, -15.0]

joints_max = [ 8.0, 28.0,
               20.0, 90.0, 90.0, 90.0,
               45.0, 20.0, 90.0, 90.0, 90.0,
               20.0, 90.0, 90.0, 90.0,
               20.0, 90.0, 90.0, 90.0,
               60.0, 70.0, 12.0, 30.0, 90.0]

joints_min_rad = np.radians(joints_min)
joints_max_rad = np.radians(joints_max)


# OpenPose callback
def jointsCB(msg: Float64MultiArray):

    # Convert to Shadow Hand
    this_joints = np.array(msg.data)

    # DEBUG
    if False:
        # Print the joint angles
        print("Received joint angles:")
        print(this_joints)

    # Check correct message
    if len(this_joints) == 24:
        
        # Check joints limits
        for i in range(0,len(this_joints)):
            if this_joints[i] > joints_max_rad[i]:
                this_joints[i] = joints_max_rad[i]
            elif this_joints[i] < joints_min_rad[i]:
                this_joints[i] = joints_min_rad[i]

        # Send joints command for Shadow Hand
        hand_pos = hand_commander.get_joints_position()

        hand_pos.update({'rh_WRJ2': this_joints[0],  'rh_WRJ1': this_joints[1]})
        hand_pos.update({'rh_FFJ4': this_joints[2],  'rh_FFJ3': this_joints[3],  'rh_FFJ2': this_joints[4],  'rh_FFJ1': this_joints[5]})
        hand_pos.update({'rh_LFJ5': this_joints[6],  'rh_LFJ4': this_joints[7],  'rh_LFJ3': this_joints[8],  'rh_LFJ2': this_joints[9],  'rh_LFJ1': this_joints[10]})
        hand_pos.update({'rh_MFJ4': this_joints[11], 'rh_MFJ3': this_joints[12], 'rh_MFJ2': this_joints[13], 'rh_MFJ1': this_joints[14]})
        hand_pos.update({'rh_RFJ4': this_joints[15], 'rh_RFJ3': this_joints[16], 'rh_RFJ2': this_joints[17], 'rh_RFJ1': this_joints[18]})
        hand_pos.update({'rh_THJ5': this_joints[19], 'rh_THJ4': this_joints[20], 'rh_THJ3': this_joints[21], 'rh_THJ2': this_joints[22], 'rh_THJ1': this_joints[23]})

        hand_commander.move_to_joint_value_target_unsafe(joint_states=hand_pos, time=0.3, wait=True, angle_degrees=False)

        print('\n' + colored('Command sent to Shadow Hand!', 'green') + '\n') 
    

if __name__ == "__main__":
    # Init ROS
    rospy.init_node('command_shadow')

    # Get ROS Parameters
    joints_topic = rospy.get_param('~joints_topic', '/shadow_joints')

    # Shadow Hand commander
    hand_commander = SrHandCommander(name='right_hand')

    # Set control velocity and acceleration
    hand_commander.set_max_velocity_scaling_factor(1.0)
    hand_commander.set_max_acceleration_scaling_factor(1.0)

    # Create subscriber
    rospy.Subscriber(joints_topic, Float64MultiArray, jointsCB, queue_size=1)

    print('\n' + colored('"command_shadow" ROS node is ready!', 'green') + '\n')  

    rospy.spin()