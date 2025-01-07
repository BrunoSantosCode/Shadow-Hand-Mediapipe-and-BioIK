#!/usr/bin/env python3

#* * * * * * * * command_ur5.py * * * * * * * *#
#*  Commands UR5 up and down with WSD keys    *#
#* * * * * * * * * * * ** * * * * * * * * * * *#

import rospy
import numpy as np
from pynput import keyboard
from termcolor import colored
from sr_robot_commander.sr_arm_commander import SrArmCommander

UR5_SPEED = 0.50

# Joints order
# ra_elbow_joint   ra_shoulder_lift_joint   ra_shoulder_pan_joint 
# ra_wrist_1_joint   ra_wrist_2_joint   ra_wrist_3_joint


# UR5 Scan Conveyor Pose
ur5_default_pose = {'ra_elbow_joint': np.radians(135), 
                    'ra_shoulder_lift_joint': np.radians(-135), 
                    'ra_shoulder_pan_joint': np.radians(-45), 
                    'ra_wrist_1_joint': np.radians(-270), 
                    'ra_wrist_2_joint': np.radians(-90), 
                    'ra_wrist_3_joint': np.radians(0)}

ur5_up_pose = {'ra_elbow_joint': np.radians(105), 
               'ra_shoulder_lift_joint': np.radians(-135), 
               'ra_shoulder_pan_joint': np.radians(-135), 
               'ra_wrist_1_joint': np.radians(-125), 
               'ra_wrist_2_joint': np.radians(-90), 
               'ra_wrist_3_joint': np.radians(0)}

ur5_down_pose = {'ra_elbow_joint': np.radians(140), 
                 'ra_shoulder_lift_joint': np.radians(-135), 
                 'ra_shoulder_pan_joint': np.radians(-135), 
                 'ra_wrist_1_joint': np.radians(-150), 
                 'ra_wrist_2_joint': np.radians(-90), 
                 'ra_wrist_3_joint': np.radians(0)}


def moveUR5to(pose: str):
    if pose == 'default':
        arm_commander.move_to_joint_value_target(ur5_default_pose, wait=True)
    elif pose == 'up':
        arm_commander.move_to_joint_value_target(ur5_up_pose, wait=True)
    elif pose == 'down':
        arm_commander.move_to_joint_value_target(ur5_down_pose, wait=True)


def on_press(key):
    if rospy.is_shutdown():
        exit()
    try:
        if key.char == 'w':
            print(colored('\nMoving UP...', 'green'))
            moveUR5to('up')
        elif key.char == 's':
            print(colored('\nMoving DOWN...', 'green'))
            moveUR5to('down')
        elif key.char == 'd':
            print(colored('\nMoving to DEFAULT pose...', 'green'))
            moveUR5to('default')
        elif key.char == 'q':
            print('\n' + colored('Exiting...', 'red'))
            return False
    except AttributeError:
        pass


if __name__ == "__main__":
    # Init ROS
    rospy.init_node('command_ur5')

    # Shadow Hand commander
    arm_commander = SrArmCommander(name='right_arm')

    # Set control velocity and acceleration
    arm_commander.set_max_velocity_scaling_factor(UR5_SPEED)
    arm_commander.set_max_acceleration_scaling_factor(UR5_SPEED)

    print('\n' + colored('"command_ur5" ROS node is ready!', 'green') + '\n')  
    print('Press W to move UP, S to move DOWN, and D to move to DEFAULT pose.')
    print('Press Q to exit.')

    # Start listening for keyboard input
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except rospy.ROSInterruptException:
            print('\n' + colored('Shutting down...', 'red'))