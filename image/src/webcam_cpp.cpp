#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "webcam_cpp");
    ros::NodeHandle nh("~");

    // Get ROS parameters
    std::string resolution_param;
    std::string image_topic;
    nh.param("camera_resolution", resolution_param, std::string("SD"));
    nh.param("image_topic", image_topic, std::string("/zed/left_image"));

    // Create ROS Publisher
    ros::Publisher imagePub = nh.advertise<sensor_msgs::Image>(image_topic, 1);

    // Create a CvBridge object to convert OpenCV images to ROS Image messages
    cv_bridge::CvImage bridge;

    // Open Webcam
    cv::VideoCapture cap(2);
    if (!cap.isOpened())
    {
        ROS_ERROR("Error: Couldn't open webcam.");
        return -1;
    }

    // Set Webcam resolution based on parameter
    if (resolution_param == "HD1080"){
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    }
    else if (resolution_param == "HD720"){
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    }
    else{
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 680);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }

    // Frame var
    cv::Mat frame;

    while (ros::ok())
    {
        // Capture frame
        cap >> frame;
        if (frame.empty())
        {
            ROS_ERROR("Error: Couldn't grab frame from webcam.");
            break;
        }

        // Convert to ROS
        bridge.image = frame;
        sensor_msgs::ImagePtr rosImage = bridge.toImageMsg();

        // Publish to ROS
        imagePub.publish(rosImage);

        // Display the frame in a window (optional)
        cv::imshow("Webcam", frame);
    }

    // Release the webcam and close OpenCV windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
