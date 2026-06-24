//
// Created by hakuroyo on 2026/5/11.
//
#include <depth_image_to_pointcloud2/depth_image_to_pointcloud2_node.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace depth_image_to_pointcloud2
{
DepthImageToPointCloud2Node::DepthImageToPointCloud2Node()
: Node("depth_image_to_pointcloud2_node")
{
    frame_stride_ = declare_parameter<int>("frame_stride", 4);
    depth_scale_ = declare_parameter<double>("depth_scale", 0.001);
    max_valid_depth_ = declare_parameter<double>("max_valid_depth", 20.0);
    show_images_ = declare_parameter<bool>("show_images", true);
    gazebo_tran_ = declare_parameter<bool>("gazebo_tran", false);

    // 闇€瑕佽皟鍙傦細杩欎簺鍙傛暟浼氱洿鎺ヤ紶缁?RGB-D 鍘婚浘绠楁硶锛屽彲鍦?launch.py 涓鐩栥€?    dehaze_params_.dark_channel_radius = declare_parameter<int>("dark_channel_radius", 7);
    dehaze_params_.omega = declare_parameter<double>("omega", 0.95);
    dehaze_params_.min_transmission = declare_parameter<double>("min_transmission", 0.10);
    dehaze_params_.atmospheric_light_percent =
        declare_parameter<double>("atmospheric_light_percent", 0.001);
    dehaze_params_.depth_compensation_strength =
        declare_parameter<double>("depth_compensation_strength", 0.35);
    dehaze_params_.max_depth_scale = declare_parameter<double>("max_depth_scale", 1.50);

    const auto qos = rclcpp::SensorDataQoS();
    rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/depth_camera/image_raw", qos,
        std::bind(&DepthImageToPointCloud2Node::rgbCallback, this, std::placeholders::_1));
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/depth_camera/depth/image_raw", qos,
        std::bind(&DepthImageToPointCloud2Node::depthCallback, this, std::placeholders::_1));
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/depth_camera/camera_info", qos,
        std::bind(&DepthImageToPointCloud2Node::cameraInfoCallback, this, std::placeholders::_1));

    pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/camera/points", 10);
    dehazed_image_pub_ = create_publisher<sensor_msgs::msg::Image>("/camera/dehazed_image", 10);

    if (show_images_ && std::getenv("DISPLAY") == nullptr) {
        RCLCPP_WARN(
            get_logger(),
            "鏈娴嬪埌 DISPLAY锛岃烦杩?OpenCV 瀹炴椂鍥惧儚绐楀彛锛涘闇€鏄剧ず绐楀彛璇蜂负瀹瑰櫒閰嶇疆 GUI 杞彂");
        show_images_ = false;
    }

    if (show_images_) {
        cv::namedWindow("RGB-D 鍘熷 RGB 鍥惧儚", cv::WINDOW_NORMAL);
        cv::namedWindow("RGB-D 鍘婚浘鍚?RGB 鍥惧儚", cv::WINDOW_NORMAL);
    }
}

void DepthImageToPointCloud2Node::rgbCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    ++frame_count_;
    if (frame_stride_ > 1 && frame_count_ % frame_stride_ != 0) {
        return;
    }
    if (!latest_depth_msg_ || !latest_camera_info_msg_) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "waiting for depth image and camera_info before publishing point cloud");
        return;
    }

    processFrame(*msg);
}

void DepthImageToPointCloud2Node::depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    latest_depth_msg_ = msg;
}

void DepthImageToPointCloud2Node::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    latest_camera_info_msg_ = msg;
}

void DepthImageToPointCloud2Node::processFrame(const sensor_msgs::msg::Image &rgb_msg)
{
    try {
        const cv_bridge::CvImagePtr rgb_cv =
            cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
        const cv::Mat depth_meters = convertDepthToMeters(*latest_depth_msg_);

        const RgbdImage dehazed = dehazeRgbdImage(
            RgbdImage{rgb_cv->image, depth_meters}, dehaze_params_);

        dehazed_image_pub_->publish(
            *cv_bridge::CvImage(
                rgb_msg.header, sensor_msgs::image_encodings::BGR8, dehazed.rgb).toImageMsg());

        if (show_images_) {
            cv::imshow("RGB-D original RGB image", rgb_cv->image);
            cv::imshow("RGB-D classical dehazed RGB image", dehazed.rgb);
            cv::waitKey(1);
        }

        auto pointcloud = buildPointCloud(
            dehazed.rgb,
            dehazed.depth,
            *latest_camera_info_msg_,
            rgb_msg.header.stamp);
        pointcloud_pub_->publish(pointcloud);
    } catch (const std::exception &error) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "RGB-D dehaze or point cloud conversion failed: %s", error.what());
    }
}

cv::Mat DepthImageToPointCloud2Node::convertDepthToMeters(
    const sensor_msgs::msg::Image &depth_msg) const
{
    cv_bridge::CvImagePtr depth_cv;
    if (depth_msg.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        depth_cv = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        return depth_cv->image.clone();
    }
    if (depth_msg.encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
        depth_msg.encoding == sensor_msgs::image_encodings::MONO16) {
        depth_cv = cv_bridge::toCvCopy(depth_msg);
        cv::Mat depth_meters;
        depth_cv->image.convertTo(depth_meters, CV_32FC1, depth_scale_);
        return depth_meters;
    }

    depth_cv = cv_bridge::toCvCopy(depth_msg);
    cv::Mat depth_meters;
    depth_cv->image.convertTo(depth_meters, CV_32FC1);
    return depth_meters;
}

sensor_msgs::msg::PointCloud2 DepthImageToPointCloud2Node::buildPointCloud(
    const cv::Mat &rgb,
    const cv::Mat &depth_meters,
    const sensor_msgs::msg::CameraInfo &camera_info,
    const rclcpp::Time &stamp) const
{
    if (rgb.empty() || depth_meters.empty()) {
        throw std::invalid_argument("RGB or Depth image is empty, cannot build point cloud");
    }
    if (rgb.size() != depth_meters.size()) {
        throw std::invalid_argument("RGB and Depth image sizes do not match, cannot build point cloud");
    }
    if (rgb.type() != CV_8UC3) {
        throw std::invalid_argument("point cloud color image must be CV_8UC3");
    }

    const double fx = camera_info.k[0] > 0.0 ? camera_info.k[0] : camera_info.p[0];
    const double fy = camera_info.k[4] > 0.0 ? camera_info.k[4] : camera_info.p[5];
    const double cx = camera_info.k[2] > 0.0 ? camera_info.k[2] : camera_info.p[2];
    const double cy = camera_info.k[5] > 0.0 ? camera_info.k[5] : camera_info.p[6];
    if (fx <= 0.0 || fy <= 0.0) {
        throw std::invalid_argument("CameraInfo is missing valid fx/fy intrinsics");
    }

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = camera_info.header.frame_id.empty() ? "camera_link" : camera_info.header.frame_id;
    cloud.height = static_cast<uint32_t>(depth_meters.rows);
    cloud.width = static_cast<uint32_t>(depth_meters.cols);
    cloud.is_bigendian = false;
    cloud.is_dense = false;
    cloud.point_step = 16;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(static_cast<size_t>(cloud.row_step) * cloud.height);

    cloud.fields.resize(4);
    cloud.fields[0].name = "x";
    cloud.fields[0].offset = 0;
    cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[0].count = 1;
    cloud.fields[1].name = "y";
    cloud.fields[1].offset = 4;
    cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[1].count = 1;
    cloud.fields[2].name = "z";
    cloud.fields[2].offset = 8;
    cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[2].count = 1;
    cloud.fields[3].name = "rgb";
    cloud.fields[3].offset = 12;
    cloud.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[3].count = 1;

    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    for (int v = 0; v < depth_meters.rows; ++v) {
        const auto *depth_row = depth_meters.ptr<float>(v);
        const auto *rgb_row = rgb.ptr<cv::Vec3b>(v);
        for (int u = 0; u < depth_meters.cols; ++u) {
            float z = depth_row[u];
            float x = quiet_nan;
            float y = quiet_nan;
            if (std::isfinite(z) && z > 0.0F && z <= static_cast<float>(max_valid_depth_)) {
                x = static_cast<float>((static_cast<double>(u) - cx) * z / fx);
                y = static_cast<float>((static_cast<double>(v) - cy) * z / fy);
                if (gazebo_tran_) {
                    const float optical_x = x;
                    const float optical_y = y;
                    const float optical_z = z;
                    x = optical_z;
                    y = -optical_x;
                    z = -optical_y;
                }
            } else {
                z = quiet_nan;
            }

            const cv::Vec3b bgr = rgb_row[u];
            const uint32_t rgb_packed =
                (static_cast<uint32_t>(bgr[2]) << 16U) |
                (static_cast<uint32_t>(bgr[1]) << 8U) |
                static_cast<uint32_t>(bgr[0]);
            float rgb_float = 0.0F;
            std::memcpy(&rgb_float, &rgb_packed, sizeof(rgb_float));

            const size_t offset =
                (static_cast<size_t>(v) * cloud.width + static_cast<size_t>(u)) * cloud.point_step;
            std::memcpy(&cloud.data[offset + 0], &x, sizeof(float));
            std::memcpy(&cloud.data[offset + 4], &y, sizeof(float));
            std::memcpy(&cloud.data[offset + 8], &z, sizeof(float));
            std::memcpy(&cloud.data[offset + 12], &rgb_float, sizeof(float));
        }
    }

    return cloud;
}

}  // namespace depth_image_to_pointcloud2
