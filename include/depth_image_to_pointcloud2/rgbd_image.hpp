//
// Shared RGB-D image data structures.
//

#ifndef ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_RGBD_IMAGE_HPP
#define ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_RGBD_IMAGE_HPP

#include <opencv2/core.hpp>

namespace depth_image_to_pointcloud2
{

struct RgbdImage
{
    cv::Mat rgb;
    cv::Mat depth;
};

}  // namespace depth_image_to_pointcloud2

#endif  // ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_RGBD_IMAGE_HPP
