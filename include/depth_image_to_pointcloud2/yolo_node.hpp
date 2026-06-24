#ifndef DEPTH_IMAGE_TO_POINTCLOUD2_YOLO_NODE_HPP
#define DEPTH_IMAGE_TO_POINTCLOUD2_YOLO_NODE_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace depth_image_to_pointcloud2
{

class YoloNode : public rclcpp::Node
{
public:
    YoloNode();

private:
    struct Detection
    {
        cv::Rect box;
        float confidence = 0.0F;
        std::uint32_t class_id = 0;
    };

    struct TargetPoint
    {
        float x = 0.0F;
        float y = 0.0F;
        float z = 0.0F;
        float intensity = 0.0F;
        std::uint32_t class_id = 0;
        float bbox_cx = 0.0F;
        float bbox_cy = 0.0F;
        float bbox_w = 0.0F;
        float bbox_h = 0.0F;
    };

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

    std::vector<Detection> detect(const cv::Mat &bgr);
    std::vector<Detection> classAwareNms(std::vector<Detection> detections) const;
    cv::Mat convertDepthToMeters(const sensor_msgs::msg::Image &depth_msg) const;
    bool detectionToTarget(
        const Detection &detection,
        const cv::Size &image_size,
        const cv::Mat &depth_meters,
        const sensor_msgs::msg::CameraInfo &camera_info,
        TargetPoint &target) const;
    sensor_msgs::msg::PointCloud2 buildDetectionCloud(
        const std::vector<TargetPoint> &targets,
        const builtin_interfaces::msg::Time &stamp) const;
    void drawDetections(cv::Mat &image, const std::vector<Detection> &detections) const;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr detection_pub_;

    sensor_msgs::msg::Image::ConstSharedPtr latest_depth_msg_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_msg_;
    cv::dnn::Net network_;

    const std::array<std::string, 6> class_names_{
        "buoy", "debris_container", "fishing_boat",
        "floating_obstacle", "platform", "vessel"};

    int input_width_ = 640;
    int input_height_ = 640;
    float confidence_threshold_ = 0.25F;
    float nms_threshold_ = 0.45F;
    double depth_scale_ = 0.001;
    double max_valid_depth_ = 100.0;
    double depth_roi_scale_ = 0.25;
    int minimum_depth_samples_ = 3;
    bool show_image_ = true;
    std::string output_frame_id_ = "base_link";
    double camera_x_ = 0.0;
    double camera_y_ = 0.0;
    double camera_z_ = 0.0;
    double camera_roll_ = 0.0;
    double camera_pitch_ = 0.0;
    double camera_yaw_ = 0.0;
};

}  // namespace depth_image_to_pointcloud2

#endif  // DEPTH_IMAGE_TO_POINTCLOUD2_YOLO_NODE_HPP
