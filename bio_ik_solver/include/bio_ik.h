/* * * * * * * * * * * * bio_ik.cpp * * * * * * * * * * * */
/*  Receives HandKeypoints.msg from "hand_kp" topic       */
/*  Uses BioIK to calculate inverse kinematics (thread1)  */
/*  Send joint angles to Shadow Hand (thread2)            */
/*  Mutex to access joint_angles                          */
/*  Adaptable median filter for keypoint positions        */
/*  Structural reorganization                             */
/*    [kp_pos -> BioIK -> angles -> Shadow Hand]          */
/*  Execute only if different angles                      */
/*  BioIK solve only if different angles keypoints        */
/*  Map human hand into Shadow Hand                       */
/*  Adjustments in hand referential                       */
/*  Sends Shadow Hand commands by SrHandCommander         */
/*  MapShadowHand after median                            */
/*  + Redefinition of Goals                               */
/* * * * * * * * * * * * * * ** * * * * * * * * * * * * * */

#include <ros/ros.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit_msgs/RobotState.h>
#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene/planning_scene.h>

#include <moveit/collision_detection/collision_matrix.h>

#include <collision_free_goal.h>
#include <bio_ik/bio_ik.h>

#include <messages/HandKeypoints.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Point.h>
#include <algorithm>
#include <vector>
#include <Eigen/Geometry>

#include <thread>
#include <mutex>
#include <chrono>

#include <iostream>
#include <fstream>

#define PI 3.14159265359
#define N_FILTER 5

// GLOBAL VARS
tf2_ros::Buffer tfBuffer, tfBuffer2;
std::string base_frame;
moveit::planning_interface::MoveGroupInterface* mgi_pointer;
const moveit::core::JointModelGroup* joint_model_group;
planning_scene::PlanningScene* planning_scene_pointer;

std::vector<std::vector<Eigen::Vector3d>> kp_positions;
std::vector<Eigen::Vector3d> prev_kp;
std::mutex mutex_kp;

bool exec_thread_started = false;
bool bio_ik_thread_started = false;

ros::Time time_begin;

// ROS RVIZ Publisher
ros::Publisher marker_pub, marker_pub_shadow, joints_shadow;


// Converts one or two geometry_msgs::Point into Eigen::Vector3d
Eigen::Vector3d point2vector(const geometry_msgs::Point& point1, const geometry_msgs::Point& point2 = geometry_msgs::Point())
{
    Eigen::Vector3d vec1(point1.x, point1.y, point1.z);
    Eigen::Vector3d vec2(point2.x, point2.y, point2.z);

    return vec1 - vec2;
}