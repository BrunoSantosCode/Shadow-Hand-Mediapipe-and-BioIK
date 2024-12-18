/* * * * * * * * bio_ik.cpp * * * * * * * */
/*  INPUT: hand keypoints                 */
/*  PROCESS: BioIK solver                 */
/*  OUTPUT: Shadow Hand joint angles      */
/* * * * * * * * * * ** * * * * * * * * * */

#include <bio_ik.h>

// Receives Hand Keypoints
void handKeypointsCB(const messages::HandKeypoints::ConstPtr& msg)
{
    // Received ROS Msg Info
    ROS_INFO("Received a Hand Keypoints message with %zu keypoints", msg->keypoints.size());

    // Start Timer
    ros::Time startTime, endTime;
    ros::Duration time;
    startTime = ros::Time::now();

    // Init Keypoint vars
    std::vector<std::vector<Eigen::Vector3d>> keypointsPos;
    std::vector<Eigen::Vector3d> prevKeypoints;
    Eigen::Vector3d empty_pos(0, 0, 0);
    std::vector<Eigen::Vector3d> empty_hand(21, empty_pos);
    for (int i=0; i<N_FILTER; i++) keypointsPos.push_back(empty_hand);
    prevKeypoints = empty_hand;

    // Get Keypoints from ROS Msg
    std::vector<Eigen::Vector3d> keypoints;
    for (int i=0; i<21; i++){
        Eigen::Vector3d auxKeypoint(msg->keypoints[i].x, msg->keypoints[i].y, msg->keypoints[i].z);
        keypoints.push_back(auxKeypoint);
    }   

    // Update keypointsPos
    keypointsPos.insert(keypointsPos.begin(), keypoints);
    keypointsPos.pop_back();

    // Ignore If Same Keypoints
    if (prevKeypoints == keypoints){
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return;
    } else {
        prevKeypoints = keypoints;
    }

    // Map Keypoints to Shadow Hand dimensions
    // No median filter applied; use keypoints directly
    std::vector<Eigen::Vector3d> linkPos(15);
    linkPos[0] = keypoints[4];   // th_tip
    linkPos[1] = keypoints[8];   // ff_tip
    linkPos[2] = keypoints[12];  // mf_tip 
    linkPos[3] = keypoints[16];  // rf_tip
    linkPos[4] = keypoints[20];  // lf_tip

    linkPos[5] = keypoints[3];   // th_distal 
    linkPos[6] = keypoints[7];   // ff_distal
    linkPos[7] = keypoints[11];  // mf_distal
    linkPos[8] = keypoints[15];  // rf_distal
    linkPos[9] = keypoints[19];  // lf_distal

    linkPos[10] = keypoints[2];  // th_middle 
    linkPos[11] = keypoints[6];  // ff_middle
    linkPos[12] = keypoints[10]; // mf_middle
    linkPos[13] = keypoints[14]; // rf_middle
    linkPos[14] = keypoints[18]; // lf_middle

    // BioIK Conditions Set
    double timeout = 0.2;
    std::vector<std::string> MapPositionlinks {
        "rh_thtip",
        "rh_fftip",
        "rh_mftip",
        "rh_rftip",
        "rh_lftip",

        "rh_thdistal",
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_lfdistal",

        "rh_thmiddle",
        "rh_ffmiddle",
        "rh_mfmiddle",
        "rh_rfmiddle",
        "rh_lfmiddle"
    };

    // BioIK Conditions Weights
    std::vector <float> MapPositionWeights {1.0,1.0,1.0,1.0,1.0, 0.75,0.25,0.25,0.25,0.25, 0.25,0.50,0.50,0.50,0.50};
    float CoupleJointsWeight = 0.75;
    float CenterJointsWeight = 0.05;
    float MinimalDisplacementWeight = 0.10;
    
    // BioIK Goals
    bio_ik::BioIKKinematicsQueryOptions ik_options;
    ik_options.replace = true;
    ik_options.return_approximate_solution = true;

    // Position Constraints
    for (int i=0; i<MapPositionlinks.size(); i++)
    {
        // rh_wrist -> base_frame
        geometry_msgs::PointStamped stamped_in;
        stamped_in.header.frame_id = "rh_wrist";
        stamped_in.point.x = linkPos[i].x();
        stamped_in.point.y = linkPos[i].y();
        stamped_in.point.z = linkPos[i].z();
        geometry_msgs::PointStamped stamped_out ;
        tfBuffer.transform(stamped_in, stamped_out, base_frame);
        tf2::Vector3 Mapposition (stamped_out.point.x, stamped_out.point.y, stamped_out.point.z);
        ik_options.goals.emplace_back(new bio_ik::PositionGoal(MapPositionlinks[i], Mapposition, MapPositionWeights[i]));
    }

    // Non-linear Shadow Hand joint coupling constraints
    auto createJointGoal = [&](const std::string& joint1, const std::string& joint2) {
        std::vector<std::string> coupled_joints {joint1, joint2};
        return new bio_ik::JointFunctionGoal(
            coupled_joints,
            [=](std::vector<double>& vv) {
                vv[0] = fmin(vv[0], vv[1]); // J1 <= J2
            }, CoupleJointsWeight);
    };
    ik_options.goals.emplace_back(createJointGoal("rh_FFJ1", "rh_FFJ2"));
    ik_options.goals.emplace_back(createJointGoal("rh_MFJ1", "rh_MFJ2"));
    ik_options.goals.emplace_back(createJointGoal("rh_RFJ1", "rh_RFJ2"));
    ik_options.goals.emplace_back(createJointGoal("rh_LFJ1", "rh_LFJ2"));

    // Center Joints Goal
    ik_options.goals.emplace_back(new bio_ik::CenterJointsGoal(CenterJointsWeight));

    // Minimal Displacement Goal
    ik_options.goals.emplace_back(new bio_ik::MinimalDisplacementGoal(MinimalDisplacementWeight));

    // Get Current Robot State
    robot_state::RobotState& current_state = (*planning_scene_pointer).getCurrentStateNonConst();   

    // RUN BioIK Solver
    bool found_ik = current_state.setFromIK(
                        joint_model_group,             // Shadow Hand joints
                        EigenSTL::vector_Isometry3d(), // no explicit poses here
                        std::vector<std::string>(),
                        timeout,
                        moveit::core::GroupStateValidityCallbackFn(),
                        ik_options
                    );

    // DEBUG
    if (false) {
        std::cout << "BioIK found solutions: " << found_ik << std::endl;
    }

    // Get Shadow Hand BioIK Joint Angles
    std::vector<double> joint_angles;
    std_msgs::Float64MultiArray joint_angles_msg;
    if (found_ik){
        current_state.copyJointGroupPositions(joint_model_group, joint_angles);
        // Set Wrist Joints to 0
        joint_angles[0] = 0;
        joint_angles[1] = 0;
        joint_angles_msg.data = joint_angles;
    } else {
        std::cout << "Did not find IK solution" << std::endl;
    }

    // DEUBG
    if (false){
        std::cout << "Sent joint angles:" << std::endl;
        for (int i=0; i<joint_angles.size(); i++)
            std::cout << " " << joint_angles[i] << " ";
        std::cout << std::endl;
    }

    // Publish Shadow Hand command
    joints_shadow.publish(joint_angles_msg);
    std::cout << "\n\033[36mJoint angles calculated!\033[0m\n" << std::endl;

    // Reset planning goals
    for (int j = 0; j <ik_options.goals.size();j++)
        ik_options.goals[j].reset();

    // Calculate BioIK solver duration
    endTime = ros::Time::now();
    time = endTime - startTime;
    std::cout << "BioIK solver duration: " << time.toNSec()/(1000000) << " ms!" << std::endl;
}

int main(int argc, char **argv)
{
    // Init ROS node
    ros::init(argc, argv, "bio_ik");
    std::cout << "\"bio_ik\" ROS node started!" << std::endl;
    ros::NodeHandle nh;

    ros::AsyncSpinner spinner(5);
    spinner.start();

    time_begin = ros::Time::now();

    // Get ROS parameters
    std::string keypoints_topic, joints_topic;
    nh.param(ros::this_node::getName() + "/keypoints_topic", keypoints_topic, std::string("/shadow_hand_keypoints"));
    nh.param(ros::this_node::getName() + "/joints_topic", joints_topic, std::string("/shadow_joints"));

    // ROS Transform
    tf2_ros::TransformListener tfListener(tfBuffer);

    // Moveit
    std::string group_name = "right_hand";
    moveit::planning_interface::MoveGroupInterface mgi(group_name);
    base_frame = mgi.getPoseReferenceFrame();

    // Set MoveGroupInterface parameters
    mgi.setGoalTolerance(0.01);
    mgi.setPlanningTime(1);
    mgi.setPlannerId("RRTConnectkConfigDefault");
    auto robot_model = mgi.getCurrentState()->getRobotModel();
    joint_model_group = robot_model->getJointModelGroup(group_name);
    moveit::core::RobotState robot_state(robot_model);
    planning_scene::PlanningScene planning_scene(robot_model);

    // DEBUG
    if (false){
        // Print joint names
        std::vector<std::string> joint_names = joint_model_group->getJointModelNames();
        for (const auto& name : joint_names) {
            std::cout << name << " ";
            std::cout << std::endl;
        }
    }

    // Set control velocity and acceleration
    mgi.setMaxAccelerationScalingFactor(1.0);
    mgi.setMaxVelocityScalingFactor(1.0);

    // Define pointers to access vars in callback
    mgi_pointer = &mgi;
    planning_scene_pointer = &planning_scene;

    // Create Subscriber
    ros::Subscriber hand_keypoints_sub = nh.subscribe(keypoints_topic, 1, handKeypointsCB);
    std::cout << "\nHand Keypoints Topic: " << keypoints_topic << std::endl;

    // Create Publisher
    joints_shadow = nh.advertise<std_msgs::Float64MultiArray>(joints_topic, 1);

    // Ready 
    std::cout << "\n\033[1;32m\"bio_ik_no_filter\" ROS node is ready!\033[0m\n" << std::endl;

    ros::waitForShutdown();
    return 0;
}
