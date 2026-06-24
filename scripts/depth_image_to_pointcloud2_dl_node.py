#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import torch

from ament_index_python.packages import get_package_share_directory

from depth_image_to_pointcloud2.dehaze_model import build_model


class DepthImageToPointCloud2DlNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_image_to_pointcloud2_node")
        self.latest_depth_msg: Image | None = None
        self.latest_camera_info_msg: CameraInfo | None = None
        self.frame_count = 0
        self.logged_first_cloud = False

        self.frame_stride = self.declare_parameter("frame_stride", 4).value
        self.depth_scale = self.declare_parameter("depth_scale", 0.001).value
        self.max_valid_depth = self.declare_parameter("max_valid_depth", 20.0).value
        self.show_images = self.declare_parameter("show_images", True).value
        self.gazebo_tran = self.as_bool(self.declare_parameter("gazebo_tran", False).value)
        self.base_channels = self.declare_parameter("dl_base_channels", 0).value

        package_share = Path(get_package_share_directory("depth_image_to_pointcloud2"))
        default_model = package_share / "models" / "best.pt"
        self.model_path = Path(
            self.declare_parameter("dl_model_path", str(default_model)).value
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(self.model_path)

        self.rgb_sub = self.create_subscription(
            Image, "/depth_camera/image_raw", self.rgb_callback, qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image, "/depth_camera/depth/image_raw", self.depth_callback, qos_profile_sensor_data
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/depth_camera/camera_info", self.camera_info_callback, qos_profile_sensor_data
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2, "/camera/points", 10)
        self.dehazed_image_pub = self.create_publisher(Image, "/camera/dehazed_image", 10)

        if self.show_images and os.environ.get("DISPLAY") is None:
            self.get_logger().warn(
                "DISPLAY is not set; disabling OpenCV image windows so point clouds can still be published"
            )
            self.show_images = False

        self.get_logger().info(
            f"DL dehaze model loaded: {self.model_path} on {self.device}; "
            f"gazebo_tran={self.gazebo_tran}"
        )

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        if not model_path.exists():
            raise FileNotFoundError(f"DL dehaze model not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        base_channels = int(self.base_channels or checkpoint.get("base_channels", 32))
        model = build_model(base_channels=base_channels).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    @staticmethod
    def as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def rgb_callback(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_stride > 1 and self.frame_count % self.frame_stride != 0:
            return
        if self.latest_depth_msg is None or self.latest_camera_info_msg is None:
            self.get_logger().warn(
                "waiting for depth image and camera_info before publishing point cloud",
                throttle_duration_sec=2.0,
            )
            return
        self.process_frame(msg)

    def depth_callback(self, msg: Image) -> None:
        self.latest_depth_msg = msg

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info_msg = msg

    def process_frame(self, rgb_msg: Image) -> None:
        try:
            bgr = self.image_msg_to_bgr8(rgb_msg)
            depth_meters = self.convert_depth_to_meters(self.latest_depth_msg)
            dehazed_bgr = self.dehaze_bgr(bgr)
            self.dehazed_image_pub.publish(self.bgr8_to_image_msg(dehazed_bgr, rgb_msg))

            if self.show_images:
                cv2.imshow("RGB-D original RGB image", bgr)
                cv2.imshow("RGB-D DL dehazed RGB image", dehazed_bgr)
                cv2.waitKey(1)

            cloud = self.build_point_cloud(
                dehazed_bgr,
                depth_meters,
                self.latest_camera_info_msg,
                rgb_msg.header.stamp,
            )
            self.pointcloud_pub.publish(cloud)
            if not self.logged_first_cloud:
                self.get_logger().info(
                    "Published first DL point cloud: "
                    f"frame_id={cloud.header.frame_id}, size={cloud.width}x{cloud.height}, "
                    f"point_step={cloud.point_step}, gazebo_tran={self.gazebo_tran}"
                )
                self.logged_first_cloud = True
        except Exception as exc:
            self.get_logger().warn(
                f"DL RGB-D dehaze or point cloud conversion failed: {exc}",
                throttle_duration_sec=2.0,
            )

    def dehaze_bgr(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).to(self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            / 255.0
        )
        with torch.no_grad():
            pred = self.model(tensor).clamp(0.0, 1.0)
        out_rgb = (
            pred.squeeze(0).permute(1, 2, 0).mul(255.0).byte().cpu().numpy()
        )
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def convert_depth_to_meters(self, depth_msg: Image) -> np.ndarray:
        depth = self.image_msg_to_array(depth_msg)
        if depth_msg.encoding == "32FC1":
            return depth.astype(np.float32)
        if depth_msg.encoding in ("16UC1", "mono16"):
            return depth.astype(np.float32) * float(self.depth_scale)
        return depth.astype(np.float32)

    @staticmethod
    def image_msg_to_array(msg: Image) -> np.ndarray:
        dtype_by_encoding = {
            "bgr8": np.uint8,
            "rgb8": np.uint8,
            "mono8": np.uint8,
            "8UC1": np.uint8,
            "16UC1": np.uint16,
            "mono16": np.uint16,
            "32FC1": np.float32,
        }
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "mono8": 1,
            "8UC1": 1,
            "16UC1": 1,
            "mono16": 1,
            "32FC1": 1,
        }
        if msg.encoding not in dtype_by_encoding:
            raise ValueError(f"unsupported image encoding: {msg.encoding}")

        dtype = np.dtype(dtype_by_encoding[msg.encoding])
        if msg.is_bigendian != (np.little_endian is False):
            dtype = dtype.newbyteorder(">")
        channels = channels_by_encoding[msg.encoding]
        array = np.frombuffer(msg.data, dtype=dtype)
        if channels == 1:
            array = array.reshape((msg.height, msg.step // dtype.itemsize))[:, : msg.width]
        else:
            row_items = msg.step // dtype.itemsize
            array = array.reshape((msg.height, row_items))[:, : msg.width * channels]
            array = array.reshape((msg.height, msg.width, channels))
        if dtype.byteorder == ">":
            array = array.byteswap().newbyteorder()
        return array.copy()

    @classmethod
    def image_msg_to_bgr8(cls, msg: Image) -> np.ndarray:
        image = cls.image_msg_to_array(msg)
        if msg.encoding == "bgr8":
            return image
        if msg.encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if msg.encoding in ("mono8", "8UC1"):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        raise ValueError(f"cannot convert {msg.encoding} to bgr8")

    @staticmethod
    def bgr8_to_image_msg(image: np.ndarray, source: Image) -> Image:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("dehazed image must be uint8 BGR")
        contiguous = np.ascontiguousarray(image)
        msg = Image()
        msg.header = source.header
        msg.height, msg.width = contiguous.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = contiguous.tobytes()
        return msg

    def build_point_cloud(
        self,
        bgr: np.ndarray,
        depth_meters: np.ndarray,
        camera_info: CameraInfo,
        stamp,
    ) -> PointCloud2:
        if bgr.shape[:2] != depth_meters.shape[:2]:
            raise ValueError("RGB and Depth image sizes do not match")

        fx = camera_info.k[0] if camera_info.k[0] > 0.0 else camera_info.p[0]
        fy = camera_info.k[4] if camera_info.k[4] > 0.0 else camera_info.p[5]
        cx = camera_info.k[2] if camera_info.k[2] > 0.0 else camera_info.p[2]
        cy = camera_info.k[5] if camera_info.k[5] > 0.0 else camera_info.p[6]
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("CameraInfo is missing valid fx/fy intrinsics")

        height, width = depth_meters.shape[:2]
        cloud = PointCloud2()
        cloud.header = camera_info.header
        cloud.header.stamp = stamp
        if not cloud.header.frame_id:
            cloud.header.frame_id = "camera_link"
        cloud.height = height
        cloud.width = width
        cloud.is_bigendian = False
        cloud.is_dense = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * cloud.width
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        data = bytearray(cloud.row_step * cloud.height)
        nan = float("nan")
        max_depth = float(self.max_valid_depth)
        for v in range(height):
            for u in range(width):
                z = float(depth_meters[v, u])
                if math.isfinite(z) and 0.0 < z <= max_depth:
                    x = (float(u) - cx) * z / fx
                    y = (float(v) - cy) * z / fy
                    if self.gazebo_tran:
                        optical_x, optical_y, optical_z = x, y, z
                        x = optical_z
                        y = -optical_x
                        z = -optical_y
                else:
                    x = y = z = nan

                blue, green, red = [int(c) for c in bgr[v, u]]
                rgb_uint32 = (red << 16) | (green << 8) | blue
                rgb_float = struct.unpack("f", struct.pack("I", rgb_uint32))[0]
                offset = (v * width + u) * cloud.point_step
                struct.pack_into("ffff", data, offset, x, y, z, rgb_float)

        cloud.data = data
        return cloud


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DepthImageToPointCloud2DlNode()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
