//
// Classical dark-channel RGB-D dehazing interface.
//

#ifndef ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_CLASSICAL_DEHAZE_HPP
#define ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_CLASSICAL_DEHAZE_HPP

#include <depth_image_to_pointcloud2/rgbd_image.hpp>

namespace depth_image_to_pointcloud2
{

struct DehazeParams
{
    int dark_channel_radius = 7;
    double omega = 0.95;
    double min_transmission = 0.10;
    double atmospheric_light_percent = 0.001;
    double depth_compensation_strength = 0.35;
    double max_depth_scale = 1.50;
};

RgbdImage dehazeRgbdImage(const RgbdImage &input, const DehazeParams &params = DehazeParams());

}  // namespace depth_image_to_pointcloud2

#endif  // ROS2_WS_DEPTH_IMAGE_TO_POINTCLOUD2_CLASSICAL_DEHAZE_HPP
