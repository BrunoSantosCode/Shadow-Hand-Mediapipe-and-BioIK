# 👌 Shadow Dexterous Hand Control via MediaPipe and BioIK Integration

This repository contains the code developed in order to teleoperate a Shadow Dexterous Hand utilising the MediaPipe and BioIK algorithms.

## 📌 Project Overview

The main purpose of this project is to teleoperate an anthropomorphic robotic hand without the need to resort to extremely expensive equipment, such as kinematic data gloves. To this end, a modular system based on the [MediaPipe](https://github.com/google-ai-edge/mediapipe) acquisition algorithm and the [BioIK](https://github.com/TAMS-Group/bio_ik) kinematic retargeting algorithm was developed and integrated. 
MediaPipe is used to acquire the position of the human hand, acquiring a total of 21 keypoints of the hand. BioIK consists of an optimisation algorithm used to map the acquired keypoints to the robotic hand, providing the angles for each one of the joints.
This repository provides the necessaary code to achieve the teleoperation using **C++**, **Python** and the **Robotic Operating System (ROS)**.

 - **Robotic Hand**: Shadow Dexterous Hand
 - **Camera**: ZED 2i stereo camera (works with webcam)
 - **Development Environment**: ROS noetic, Docker

## 🎥 Watch Shadow Hand in Action
 To see the Shadow Hand grasping different objects controlled by teleoperation, check out the **YouTube demo video**:  
 [Watch the video here](https://youtu.be/_mvc2GK_sRg)

## 🗂️ Folder Structure
 - **[`image`](image)**: ROS package responsible for the image acquisition process. This package should be placed in the Camera docker container at `/root/catkin_ws/src`;
 - **[`mediapipe`](mediapipe)**: ROS package runnning MediaPipe. This package should be placed in the Camera docker container at `/root/catkin_ws/src`;
 - **[`bio_ik_solver`](bio_ik_solver)**: ROS package containing the BioIK inverse kinematics solver code. This package should be placed in the Shadow Hand docker container at `/home/user/projects/shadow_robot/base/src`;
 - **[`shadow_control`](shadow_control)**: ROS package responsible for sending the control commands to Shadow Hand. This package should be placed in the Shadow Hand docker container at `/home/user/projects/shadow_robot/base/src`;
 - **[`messages`](messages)**: ROS package containing the ROS message responsible to send the hand keypoints positions from the MediaPipe module to the BioIK module. This package should be placed both in the Shadow Hand docker container at `/home/user/projects/shadow_robot/base/src` and in the Camera docker container at `/root/catkin_ws/src`.


## ⚙️ Main Scripts
 - [`webcam_cpp.cpp`](image/src/webcam_cpp.cpp): acquire images from webcam;
 - [`zed_cpp.cpp`](image/src/zed_cpp.cpp): acquire images from ZED 2i camera (both left and right images);
 - [`mediapipe_2d.py`](mediapipe/src/mediapipe_2d.py): runs MediaPipe, extracting 3D hand keypoints;
 - [`mediapipe_stereo.py`](mediapipe/src/mediapipe_stereo.py): runs two instances of MediaPipe, for left and right ZED camera images and the calculates the 3D hand keypoints based on stereo vision techniques;
 - [`bio_ik.py`](bio_ik_solver/src/bio_ik.cpp): runs the kinematic retargeting algorithm BioIK;
 - [`command_shadow.py`](shadow_control/src/command_shadow.py): sends joint angles to Shadow Hand.

## 🚀 How to Run

1. Turn on Shadow Dexterous Hand
   
2. Execute `Launch Shadow Right Hand and Arm.desktop`

3. In `Server Docker Container` terminal run `bio_ik_solver.py`
    ```bash
      roslaunch bio_ik_solver bio_ik.launch
    ```
    
4. In `Server Docker Container`, open a new tab and run `command_shadow.py`
    ```bash
      roslaunch shadow_control command_shadow.launch
    ```

5. Open Camera container

6. In `Camera` container run `webcam_cpp.cpp` **OR** `zed_cpp.cpp`
    ```bash
      roslaunch image webcam_cpp.launch
    ```
    **OR**
    ```bash
      roslaunch image zed_cpp.launch
    ```
    
8. In `Camera`, open a new tab and run `mediapipe_2d.py` **OR** `mediapipe_stereo.py`
    ```bash
      roslaunch mediapipe mediapipe_2d.launch
    ```
    **OR**
    ```bash
      roslaunch mediapipe mediapipe_stereo.launch
    ```
    
📝 Note: MediaPipe stereo mode only works with ZED camera.
    

## 📄 Research Paper
A detailed explanation of the project, methodologies, and results will be available in an upcoming paper, which is currently TO BE PUBLISHED. Stay tuned for updates!
    
## 📫 Contact

Developed by Bruno Santos in DIGI2 Lab

Feel free to reach out via email: brunosantos@fe.up.pt

Last updated in: ``15/01/2025``

