#include <depth_image_to_pointcloud2/yolo_node.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace depth_image_to_pointcloud2
{
namespace
{

constexpr char kDetectionWindow[] = "YOLO detections on dehazed image";

float intersectionOverUnion(const cv::Rect &a, const cv::Rect &b)
{
    const cv::Rect intersection = a & b;
    const float intersection_area = static_cast<float>(intersection.area());
    const float union_area = static_cast<float>(a.area() + b.area() - intersection.area());
    return union_area > 0.0F ? intersection_area / union_area : 0.0F;
}

}  // namespace

YoloNode::YoloNode()
: Node("yolo_node")
{
    const std::string package_share =
        ament_index_cpp::get_package_share_directory("depth_image_to_pointcloud2");
    const std::string default_model_path = package_share + "/yolo/best.onnx";

    const std::string model_path = declare_parameter<std::string>("model_path", default_model_path);
    input_width_ = declare_parameter<int>("input_width", 640);
    input_height_ = declare_parameter<int>("input_height", 640);
    confidence_threshold_ =
        static_cast<float>(declare_parameter<double>("confidence_threshold", 0.25));
    nms_threshold_ = static_cast<float>(declare_parameter<double>("nms_threshold", 0.45));
    depth_scale_ = declare_parameter<double>("depth_scale", 0.001);
    max_valid_depth_ = declare_parameter<double>("max_valid_depth", 100.0);
    depth_roi_scale_ = declare_parameter<double>("depth_roi_scale", 0.25);
    minimum_depth_samples_ = declare_parameter<int>("minimum_depth_samples", 3);
    show_image_ = declare_parameter<bool>("show_image", true);
    output_frame_id_ = declare_parameter<std::string>("output_frame_id", "base_link");
    camera_x_ = declare_parameter<double>("camera_x", 0.0);
    camera_y_ = declare_parameter<double>("camera_y", 0.0);
    camera_z_ = declare_parameter<double>("camera_z", 0.0);
    camera_roll_ = declare_parameter<double>("camera_roll", 0.0);
    camera_pitch_ = declare_parameter<double>("camera_pitch", 0.0);
    camera_yaw_ = declare_parameter<double>("camera_yaw", 0.0);

    if (input_width_ <= 0 || input_height_ <= 0) {
        throw std::invalid_argument("YOLO input_width and input_height must be positive");
    }
    if (confidence_threshold_ < 0.0F || confidence_threshold_ > 1.0F) {
        throw std::invalid_argument("confidence_threshold must be in [0, 1]");
    }
    if (depth_roi_scale_ <= 0.0 || depth_roi_scale_ > 1.0) {
        throw std::invalid_argument("depth_roi_scale must be in (0, 1]");
    }

    try {
        network_ = cv::dnn::readNetFromONNX(model_path);
        network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception &error) {
        throw std::runtime_error(
                  "failed to load YOLO ONNX model '" + model_path + "': " + error.what());
    }

    const auto qos = rclcpp::SensorDataQoS();
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera/dehazed_image", qos,
        std::bind(&YoloNode::imageCallback, this, std::placeholders::_1));
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/depth_camera/depth/image_raw", qos,
        std::bind(&YoloNode::depthCallback, this, std::placeholders::_1));
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/depth_camera/camera_info", qos,
        std::bind(&YoloNode::cameraInfoCallback, this, std::placeholders::_1));
    detection_pub_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/camara/topic_xxx", 10);

    if (show_image_ && std::getenv("DISPLAY") == nullptr) {
        RCLCPP_WARN(get_logger(), "DISPLAY is not set; disabling the YOLO OpenCV window");
        show_image_ = false;
    }
    if (show_image_) {
        cv::namedWindow(kDetectionWindow, cv::WINDOW_NORMAL);
    }

    RCLCPP_INFO(
        get_logger(),
        "YOLO model loaded: %s; input=%dx%d, confidence=%.2f, output=/camara/topic_xxx",
        model_path.c_str(), input_width_, input_height_, confidence_threshold_);
}

void YoloNode::depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    latest_depth_msg_ = msg;
}

void YoloNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    latest_camera_info_msg_ = msg;
}

void YoloNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    if (!latest_depth_msg_ || !latest_camera_info_msg_) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "waiting for depth image and camera_info before publishing YOLO detections");
        return;
    }

    try {
        const cv_bridge::CvImageConstPtr image =
            cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        const cv::Mat depth_meters = convertDepthToMeters(*latest_depth_msg_);
        const std::vector<Detection> detections = detect(image->image);

        std::vector<TargetPoint> targets;
        targets.reserve(detections.size());
        for (const Detection &detection : detections) {
            TargetPoint target;
            if (detectionToTarget(
                    detection, image->image.size(), depth_meters,
                    *latest_camera_info_msg_, target)) {
                targets.push_back(target);
            }
        }

        detection_pub_->publish(buildDetectionCloud(targets, msg->header.stamp));

        if (show_image_) {
            cv::Mat visualization = image->image.clone();
            drawDetections(visualization, detections);
            cv::imshow(kDetectionWindow, visualization);
            cv::waitKey(1);
        }
    } catch (const std::exception &error) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "YOLO inference or detection point conversion failed: %s", error.what());
    }
}

std::vector<YoloNode::Detection> YoloNode::detect(const cv::Mat &bgr)
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("YOLO input image must be a non-empty BGR8 image");
    }

    const float scale = std::min(
        static_cast<float>(input_width_) / static_cast<float>(bgr.cols),
        static_cast<float>(input_height_) / static_cast<float>(bgr.rows));
    const int resized_width = static_cast<int>(std::round(bgr.cols * scale));
    const int resized_height = static_cast<int>(std::round(bgr.rows * scale));
    const int pad_x = (input_width_ - resized_width) / 2;
    const int pad_y = (input_height_ - resized_height) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height));
    cv::Mat letterboxed(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_width, resized_height)));

    const cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed, 1.0 / 255.0, cv::Size(input_width_, input_height_),
        cv::Scalar(), true, false, CV_32F);
    network_.setInput(blob);

    std::vector<cv::Mat> outputs;
    network_.forward(outputs, network_.getUnconnectedOutLayersNames());
    if (outputs.empty()) {
        throw std::runtime_error("unexpected YOLO output tensor");
    }

    std::vector<Detection> candidates;
    const cv::Rect image_bounds(0, 0, bgr.cols, bgr.rows);
    const auto append_candidate = [
        &candidates, &image_bounds, pad_x, pad_y, scale
    ](float center_x, float center_y, float width, float height,
      float confidence, std::uint32_t class_id) {
        center_x = (center_x - static_cast<float>(pad_x)) / scale;
        center_y = (center_y - static_cast<float>(pad_y)) / scale;
        width /= scale;
        height /= scale;
        cv::Rect box(
            static_cast<int>(std::round(center_x - width * 0.5F)),
            static_cast<int>(std::round(center_y - height * 0.5F)),
            static_cast<int>(std::round(width)),
            static_cast<int>(std::round(height)));
        box &= image_bounds;
        if (box.width > 0 && box.height > 0) {
            candidates.push_back(Detection{box, confidence, class_id});
        }
    };

    // The bundled ONNX exposes raw YOLOv8 DFL box logits and class logits.
    // Keeping the dynamic DFL decode in C++ avoids shape operators unsupported by
    // the OpenCV 4.5 version shipped with ROS 2 Humble on Ubuntu 22.04.
    const cv::Mat *box_logits = nullptr;
    const cv::Mat *class_logits = nullptr;
    for (const cv::Mat &output : outputs) {
        if (output.dims != 3) {
            continue;
        }
        if (output.size[1] == 64) {
            box_logits = &output;
        } else if (output.size[1] == static_cast<int>(class_names_.size())) {
            class_logits = &output;
        }
    }

    if (box_logits != nullptr && class_logits != nullptr &&
        box_logits->size[2] == class_logits->size[2]) {
        constexpr int kDistributionBins = 16;
        const cv::Mat box_rows = box_logits->reshape(1, box_logits->size[1]);
        const cv::Mat class_rows = class_logits->reshape(1, class_logits->size[1]);
        const int anchor_count = box_logits->size[2];

        int level_start = 0;
        int stride = 8;
        int grid_width = input_width_ / stride;
        int grid_height = input_height_ / stride;
        for (int anchor = 0; anchor < anchor_count; ++anchor) {
            while (anchor >= level_start + grid_width * grid_height && stride < 32) {
                level_start += grid_width * grid_height;
                stride *= 2;
                grid_width = input_width_ / stride;
                grid_height = input_height_ / stride;
            }
            const int level_index = anchor - level_start;
            const int grid_x = level_index % grid_width;
            const int grid_y = level_index / grid_width;

            float confidence = 0.0F;
            std::uint32_t class_id = 0;
            for (std::size_t class_index = 0; class_index < class_names_.size(); ++class_index) {
                const float logit = class_rows.at<float>(static_cast<int>(class_index), anchor);
                const float score = 1.0F / (1.0F + std::exp(-logit));
                if (score > confidence) {
                    confidence = score;
                    class_id = static_cast<std::uint32_t>(class_index);
                }
            }
            if (confidence < confidence_threshold_) {
                continue;
            }

            std::array<float, 4> distances{};
            for (int side = 0; side < 4; ++side) {
                float max_logit = -std::numeric_limits<float>::infinity();
                for (int bin = 0; bin < kDistributionBins; ++bin) {
                    max_logit = std::max(
                        max_logit,
                        box_rows.at<float>(side * kDistributionBins + bin, anchor));
                }
                float weight_sum = 0.0F;
                float weighted_bin_sum = 0.0F;
                for (int bin = 0; bin < kDistributionBins; ++bin) {
                    const float weight = std::exp(
                        box_rows.at<float>(side * kDistributionBins + bin, anchor) - max_logit);
                    weight_sum += weight;
                    weighted_bin_sum += weight * static_cast<float>(bin);
                }
                distances[side] = weighted_bin_sum / weight_sum;
            }

            const float anchor_x = static_cast<float>(grid_x) + 0.5F;
            const float anchor_y = static_cast<float>(grid_y) + 0.5F;
            const float center_x =
                (anchor_x + (distances[2] - distances[0]) * 0.5F) * stride;
            const float center_y =
                (anchor_y + (distances[3] - distances[1]) * 0.5F) * stride;
            const float width = (distances[0] + distances[2]) * stride;
            const float height = (distances[1] + distances[3]) * stride;
            append_candidate(center_x, center_y, width, height, confidence, class_id);
        }
    } else if (outputs.size() == 1U && outputs.front().dims == 3) {
        // Also accept a standard fully decoded Ultralytics output [1, 10, N].
        const cv::Mat &output = outputs.front();
        const int attribute_count = 4 + static_cast<int>(class_names_.size());
        cv::Mat rows;
        if (output.size[1] == attribute_count) {
            const cv::Mat attributes = output.reshape(1, output.size[1]);
            cv::transpose(attributes, rows);
        } else if (output.size[2] == attribute_count) {
            rows = output.reshape(1, output.size[1]);
        } else {
            throw std::runtime_error("unexpected decoded YOLO output shape");
        }

        for (int row_index = 0; row_index < rows.rows; ++row_index) {
            const float *row = rows.ptr<float>(row_index);
            const float *class_begin = row + 4;
            const float *class_end = class_begin + class_names_.size();
            const auto best_class = std::max_element(class_begin, class_end);
            const float confidence = *best_class;
            if (confidence < confidence_threshold_) {
                continue;
            }
            append_candidate(
                row[0], row[1], row[2], row[3], confidence,
                static_cast<std::uint32_t>(std::distance(class_begin, best_class)));
        }
    } else {
        throw std::runtime_error("unexpected YOLO output shapes");
    }

    if (box_logits != nullptr && box_logits->size[2] !=
        (input_width_ / 8) * (input_height_ / 8) +
        (input_width_ / 16) * (input_height_ / 16) +
        (input_width_ / 32) * (input_height_ / 32)) {
        throw std::runtime_error("raw YOLO anchor count does not match configured input size");
    }

    return classAwareNms(std::move(candidates));
}

std::vector<YoloNode::Detection> YoloNode::classAwareNms(
    std::vector<Detection> detections) const
{
    std::sort(
        detections.begin(), detections.end(),
        [](const Detection &left, const Detection &right) {
            return left.confidence > right.confidence;
        });

    std::vector<Detection> kept;
    for (const Detection &candidate : detections) {
        const bool suppressed = std::any_of(
            kept.begin(), kept.end(),
            [this, &candidate](const Detection &selected) {
                return selected.class_id == candidate.class_id &&
                       intersectionOverUnion(selected.box, candidate.box) > nms_threshold_;
            });
        if (!suppressed) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

cv::Mat YoloNode::convertDepthToMeters(const sensor_msgs::msg::Image &depth_msg) const
{
    cv_bridge::CvImagePtr depth;
    if (depth_msg.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        return depth->image.clone();
    }
    if (depth_msg.encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
        depth_msg.encoding == sensor_msgs::image_encodings::MONO16) {
        depth = cv_bridge::toCvCopy(depth_msg);
        cv::Mat meters;
        depth->image.convertTo(meters, CV_32FC1, depth_scale_);
        return meters;
    }

    depth = cv_bridge::toCvCopy(depth_msg);
    cv::Mat meters;
    depth->image.convertTo(meters, CV_32FC1);
    return meters;
}

bool YoloNode::detectionToTarget(
    const Detection &detection,
    const cv::Size &image_size,
    const cv::Mat &depth_meters,
    const sensor_msgs::msg::CameraInfo &camera_info,
    TargetPoint &target) const
{
    const double bbox_center_x = detection.box.x + detection.box.width * 0.5;
    const double bbox_center_y = detection.box.y + detection.box.height * 0.5;
    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    target.x = quiet_nan;
    target.y = quiet_nan;
    target.z = quiet_nan;
    target.intensity = detection.confidence;
    target.class_id = detection.class_id;
    target.bbox_cx = static_cast<float>(bbox_center_x);
    target.bbox_cy = static_cast<float>(bbox_center_y);
    target.bbox_w = static_cast<float>(detection.box.width);
    target.bbox_h = static_cast<float>(detection.box.height);

    // Preserve one PointCloud2 row per detection. If depth or intrinsics are
    // unavailable, only x/y/z remain NaN while class, confidence and box stay valid.
    if (depth_meters.empty() || depth_meters.type() != CV_32FC1) {
        return true;
    }

    const double image_to_depth_x =
        static_cast<double>(depth_meters.cols) / static_cast<double>(image_size.width);
    const double image_to_depth_y =
        static_cast<double>(depth_meters.rows) / static_cast<double>(image_size.height);
    const double depth_center_x = bbox_center_x * image_to_depth_x;
    const double depth_center_y = bbox_center_y * image_to_depth_y;

    const int roi_width = std::max(
        1, static_cast<int>(std::round(detection.box.width * image_to_depth_x * depth_roi_scale_)));
    const int roi_height = std::max(
        1, static_cast<int>(std::round(detection.box.height * image_to_depth_y * depth_roi_scale_)));
    cv::Rect depth_roi(
        static_cast<int>(std::round(depth_center_x)) - roi_width / 2,
        static_cast<int>(std::round(depth_center_y)) - roi_height / 2,
        roi_width, roi_height);
    depth_roi &= cv::Rect(0, 0, depth_meters.cols, depth_meters.rows);

    std::vector<float> valid_depths;
    valid_depths.reserve(static_cast<std::size_t>(depth_roi.area()));
    for (int row = depth_roi.y; row < depth_roi.y + depth_roi.height; ++row) {
        const float *depth_row = depth_meters.ptr<float>(row);
        for (int col = depth_roi.x; col < depth_roi.x + depth_roi.width; ++col) {
            const float value = depth_row[col];
            if (std::isfinite(value) && value > 0.0F &&
                value <= static_cast<float>(max_valid_depth_)) {
                valid_depths.push_back(value);
            }
        }
    }
    if (valid_depths.size() < static_cast<std::size_t>(std::max(1, minimum_depth_samples_))) {
        return true;
    }

    const auto median = valid_depths.begin() + valid_depths.size() / 2;
    std::nth_element(valid_depths.begin(), median, valid_depths.end());
    const double forward = *median;

    double fx = camera_info.k[0] > 0.0 ? camera_info.k[0] : camera_info.p[0];
    double fy = camera_info.k[4] > 0.0 ? camera_info.k[4] : camera_info.p[5];
    double cx = camera_info.k[2] > 0.0 ? camera_info.k[2] : camera_info.p[2];
    double cy = camera_info.k[5] > 0.0 ? camera_info.k[5] : camera_info.p[6];
    if (fx <= 0.0 || fy <= 0.0) {
        return true;
    }
    if (camera_info.width > 0U && camera_info.height > 0U) {
        const double intrinsics_scale_x =
            static_cast<double>(depth_meters.cols) / camera_info.width;
        const double intrinsics_scale_y =
            static_cast<double>(depth_meters.rows) / camera_info.height;
        fx *= intrinsics_scale_x;
        cx *= intrinsics_scale_x;
        fy *= intrinsics_scale_y;
        cy *= intrinsics_scale_y;
    }

    // Optical frame (right, down, forward) to aligned body frame (forward, left, up).
    const double body_x = forward;
    const double body_y = -(depth_center_x - cx) * forward / fx;
    const double body_z = -(depth_center_y - cy) * forward / fy;

    const double cr = std::cos(camera_roll_);
    const double sr = std::sin(camera_roll_);
    const double cp = std::cos(camera_pitch_);
    const double sp = std::sin(camera_pitch_);
    const double cyaw = std::cos(camera_yaw_);
    const double syaw = std::sin(camera_yaw_);
    const double rotated_x =
        cyaw * cp * body_x + (cyaw * sp * sr - syaw * cr) * body_y +
        (cyaw * sp * cr + syaw * sr) * body_z;
    const double rotated_y =
        syaw * cp * body_x + (syaw * sp * sr + cyaw * cr) * body_y +
        (syaw * sp * cr - cyaw * sr) * body_z;
    const double rotated_z = -sp * body_x + cp * sr * body_y + cp * cr * body_z;

    target.x = static_cast<float>(rotated_x + camera_x_);
    target.y = static_cast<float>(rotated_y + camera_y_);
    target.z = static_cast<float>(rotated_z + camera_z_);
    return true;
}

sensor_msgs::msg::PointCloud2 YoloNode::buildDetectionCloud(
    const std::vector<TargetPoint> &targets,
    const builtin_interfaces::msg::Time &stamp) const
{
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = output_frame_id_;
    cloud.height = 1;
    cloud.width = static_cast<std::uint32_t>(targets.size());
    cloud.is_bigendian = false;
    cloud.is_dense = false;
    cloud.point_step = 36;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);

    const std::array<const char *, 9> field_names{
        "x", "y", "z", "intensity", "class_id",
        "bbox_cx", "bbox_cy", "bbox_w", "bbox_h"};
    cloud.fields.resize(field_names.size());
    for (std::size_t index = 0; index < field_names.size(); ++index) {
        cloud.fields[index].name = field_names[index];
        cloud.fields[index].offset = static_cast<std::uint32_t>(index * 4U);
        cloud.fields[index].datatype = index == 4U ?
            sensor_msgs::msg::PointField::UINT32 : sensor_msgs::msg::PointField::FLOAT32;
        cloud.fields[index].count = 1;
    }

    for (std::size_t index = 0; index < targets.size(); ++index) {
        const TargetPoint &target = targets[index];
        const std::size_t offset = index * cloud.point_step;
        std::memcpy(&cloud.data[offset + 0], &target.x, sizeof(float));
        std::memcpy(&cloud.data[offset + 4], &target.y, sizeof(float));
        std::memcpy(&cloud.data[offset + 8], &target.z, sizeof(float));
        std::memcpy(&cloud.data[offset + 12], &target.intensity, sizeof(float));
        std::memcpy(&cloud.data[offset + 16], &target.class_id, sizeof(std::uint32_t));
        std::memcpy(&cloud.data[offset + 20], &target.bbox_cx, sizeof(float));
        std::memcpy(&cloud.data[offset + 24], &target.bbox_cy, sizeof(float));
        std::memcpy(&cloud.data[offset + 28], &target.bbox_w, sizeof(float));
        std::memcpy(&cloud.data[offset + 32], &target.bbox_h, sizeof(float));
    }
    return cloud;
}

void YoloNode::drawDetections(
    cv::Mat &image, const std::vector<Detection> &detections) const
{
    const cv::Scalar red(0, 0, 255);
    for (const Detection &detection : detections) {
        cv::rectangle(image, detection.box, red, 2);
        const std::string label =
            class_names_.at(detection.class_id) + " " +
            cv::format("%.2f", detection.confidence);
        int baseline = 0;
        const cv::Size label_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int text_x = detection.box.x;
        const int text_y = std::max(label_size.height, detection.box.y - 4);
        cv::putText(
            image, label, cv::Point(text_x, text_y),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, red, 1, cv::LINE_AA);
    }
}

}  // namespace depth_image_to_pointcloud2

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<depth_image_to_pointcloud2::YoloNode>());
    } catch (const std::exception &error) {
        RCLCPP_FATAL(rclcpp::get_logger("yolo_node"), "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
