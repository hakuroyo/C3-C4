//
// Created by hakuroyo on 2026/5/11.
//

#ifndef ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_NODE_HPP
#define ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_NODE_HPP

#include <depth_image_to_pointcloud2/classical_dehaze.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace depth_image_to_pointcloud2
{

class DepthImageToPointCloud2Node : public rclcpp::Node
{
public:
    DepthImageToPointCloud2Node();

private:
    void rgbCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void processFrame(const sensor_msgs::msg::Image &rgb_msg);

    cv::Mat convertDepthToMeters(const sensor_msgs::msg::Image &depth_msg) const;
    sensor_msgs::msg::PointCloud2 buildPointCloud(
        const cv::Mat &rgb,
        const cv::Mat &depth_meters,
        const sensor_msgs::msg::CameraInfo &camera_info,
        const rclcpp::Time &stamp) const;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr dehazed_image_pub_;

    sensor_msgs::msg::Image::ConstSharedPtr latest_depth_msg_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_msg_;

    DehazeParams dehaze_params_;
    int frame_stride_ = 4;
    int frame_count_ = 0;
    double depth_scale_ = 0.001;
    double max_valid_depth_ = 20.0;
    bool show_images_ = true;
    bool gazebo_tran_ = false;
};

}  // namespace depth_image_to_pointcloud2

#endif //ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_NODE_HPP
