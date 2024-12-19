#include <ros/ros.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv) {
    
    // Init ROS
    ros::init(argc, argv, "zed_cpp");
    ros::NodeHandle nh("~");

    // Get ROS Parameters
    std::string resolution_param, depth_mode_param;
    std::string left_topic_param, right_topic_param, stereo_topic_param;
    nh.param<std::string>("camera_resolution", resolution_param, "HD");
    nh.param<std::string>("depth_mode", depth_mode_param, "PERFORMANCE");
    nh.param<std::string>("left_image_topic", left_topic_param, "/zed/left_image");
    //nh.param<std::string>("right_image_topic", right_topic_param, "/zed/right_image");
    nh.param<std::string>("stereo_image_topic", stereo_topic_param, "/zed/stereo_image");

    // Map resolution parameter to ZED SDK resolution
    sl::RESOLUTION camera_resolution;
    if (resolution_param == "VGA") {
        camera_resolution = sl::RESOLUTION::VGA;
    } else if (resolution_param == "HD") {
        camera_resolution = sl::RESOLUTION::HD720;
    } else if (resolution_param == "FHD") {
        camera_resolution = sl::RESOLUTION::HD1080;
    } else {
        camera_resolution = sl::RESOLUTION::HD2K;
    }

    // Map depth mode parameter to ZED SDK depth mode
    sl::DEPTH_MODE depth_mode;
    if (depth_mode_param == "PERFORMANCE") {
        depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    } else if (depth_mode_param == "QUALITY") {
        depth_mode = sl::DEPTH_MODE::QUALITY;
    } else if (depth_mode_param == "ULTRA") {
        depth_mode = sl::DEPTH_MODE::ULTRA;
    } else {
        depth_mode = sl::DEPTH_MODE::NEURAL;
    }

    // Create ROS Publishers
    ros::Publisher leftImagePub = nh.advertise<sensor_msgs::Image>(left_topic_param, 1);
    //ros::Publisher rightImagePub = nh.advertise<sensor_msgs::Image>(right_topic_param, 1);
    ros::Publisher stereoImagePub = nh.advertise<sensor_msgs::Image>(stereo_topic_param, 1);

    // Create ZED camera object
    sl::Camera zed;

    // Configure initialization parameters
    sl::InitParameters init_params;
    init_params.camera_resolution = camera_resolution;
    init_params.depth_mode = depth_mode;

    // Open the ZED camera
    sl::ERROR_CODE status = zed.open(init_params);
    if (status != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Camera Open: " << status << ". Exiting program." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set runtime parameters
    sl::RuntimeParameters runtime_params;

    // ZED image variables
    sl::Mat leftImage, rightImage;
    cv::Mat cvLeftImage, cvRightImage, cvStereoImage;

    // ROS message converter
    cv_bridge::CvImage leftBridge, rightBridge, stereoBridge;
    leftBridge.encoding = "bgr8";
    rightBridge.encoding = "bgr8";
    stereoBridge.encoding = "bgr8";

    while (ros::ok()) {

        // Grab an image
        if (zed.grab(runtime_params) == sl::ERROR_CODE::SUCCESS) {

            // Capture ZED data
            zed.retrieveImage(leftImage, sl::VIEW::LEFT);
            zed.retrieveImage(rightImage, sl::VIEW::RIGHT);

            // Convert to OpenCV
            cvLeftImage = cv::Mat(leftImage.getHeight(), leftImage.getWidth(), CV_8UC4, leftImage.getPtr<sl::uchar1>(sl::MEM::CPU));
            cv::cvtColor(cvLeftImage, cvLeftImage, cv::COLOR_BGRA2BGR);
            cvRightImage = cv::Mat(rightImage.getHeight(), rightImage.getWidth(), CV_8UC4, rightImage.getPtr<sl::uchar1>(sl::MEM::CPU));
            cv::cvtColor(cvRightImage, cvRightImage, cv::COLOR_BGRA2BGR);
            cv::hconcat(cvLeftImage, cvRightImage, cvStereoImage);

            // Convert to ROS
            leftBridge.image = cvLeftImage;
            //rightBridge.image = cvRightImage;
            stereoBridge.image = cvStereoImage;
            sensor_msgs::ImagePtr rosLeftImage = leftBridge.toImageMsg();
            //sensor_msgs::ImagePtr rosRightImage = rightBridge.toImageMsg();
            sensor_msgs::ImagePtr rosStereoImage = stereoBridge.toImageMsg();

            // Publish to ROS
            leftImagePub.publish(rosLeftImage);
            //rightImagePub.publish(rosRightImage);
            stereoImagePub.publish(rosStereoImage);
        }
    }

    // Close ZED camera
    zed.close();
}