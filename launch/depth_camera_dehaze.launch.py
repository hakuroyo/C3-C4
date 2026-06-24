from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import PythonExpression
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = FindPackageShare("depth_image_to_pointcloud2")

    world_file = PathJoinSubstitution([package_share, "worlds", "fog_depth_camera.world"])
    rviz_config = PathJoinSubstitution([package_share, "rviz", "pointcloud.rviz"])
    gazebo_tran = ParameterValue(LaunchConfiguration("gazebo_tran"), value_type=bool)
    show_images = ParameterValue(LaunchConfiguration("show_images"), value_type=bool)

    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="true",
        description="Set to false to skip gzclient.",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Set to true to start RViz.",
    )
    gazebo_arg = DeclareLaunchArgument(
        "gazebo",
        default_value="false",
        description="Set to true to start Gazebo simulation.",
    )
    show_images_arg = DeclareLaunchArgument(
        "show_images",
        default_value="true",
        description="Set to false to disable OpenCV image windows.",
    )
    gazebo_tran_arg = DeclareLaunchArgument(
        "gazebo_tran",
        default_value="false",
        description=(
            "Transform point cloud coordinates from Gazebo optical frame "
            "(X right, Y down, Z forward) to ROS/RViz standard frame "
            "(X forward, Y left, Z up)."
        ),
    )
    dl_arg = DeclareLaunchArgument(
        "dl",
        default_value="false",
        description="Set to true to use the deep-learning dehaze model from models/.",
    )
    yolo_arg = DeclareLaunchArgument(
        "yolo",
        default_value="true",
        description="Set to true to run YOLO on the dehazed camera image.",
    )

    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver",
            world_file,
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
            "-s",
            "libgazebo_ros_force_system.so",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("gazebo")),
    )

    gazebo_client = ExecuteProcess(
        cmd=["gzclient"],
        output="screen",
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    LaunchConfiguration("gazebo"),
                    "' == 'true' and '",
                    LaunchConfiguration("gui"),
                    "' == 'true'",
                ]
            )
        ),
    )

    rgbd_processor = Node(
        package="depth_image_to_pointcloud2",
        executable="depth_image_to_pointcloud2",
        name="depth_image_to_pointcloud2",
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("dl")),
        parameters=[
            {
                # Process one frame out of every 4 to reduce load.
                "frame_stride": 4,
                # 16UC1 depth scale; Gazebo 32FC1 depth is already in meters.
                "depth_scale": 0.001,
                # Points beyond this distance are published as NaN.
                "max_valid_depth": 20.0,
                "show_images": show_images,
                # Classical dehaze parameters.
                "dark_channel_radius": 7,
                "omega": 0.95,
                "min_transmission": 0.10,
                "atmospheric_light_percent": 0.001,
                "depth_compensation_strength": 0.35,
                "max_depth_scale": 1.50,
                "gazebo_tran": gazebo_tran,
            }
        ],
    )
    rgbd_processor_dl = Node(
        package="depth_image_to_pointcloud2",
        executable="depth_image_to_pointcloud2_dl",
        name="depth_image_to_pointcloud2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("dl")),
        parameters=[
            {
                "frame_stride": 4,
                "depth_scale": 0.001,
                "max_valid_depth": 20.0,
                "show_images": show_images,
                "gazebo_tran": gazebo_tran,
            }
        ],
    )
    yolo_detector = Node(
        package="depth_image_to_pointcloud2",
        executable="yolo_node",
        name="yolo_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("yolo")),
        parameters=[
            {
                "confidence_threshold": 0.25,
                "nms_threshold": 0.45,
                "depth_scale": 0.001,
                "max_valid_depth": 100.0,
                "show_image": show_images,
                "output_frame_id": "base_link",
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    delayed_visualization = TimerAction(
        period=5.0,
        actions=[gazebo_client, rgbd_processor, rgbd_processor_dl, yolo_detector, rviz],
    )

    return LaunchDescription(
        [
            gui_arg,
            rviz_arg,
            gazebo_arg,
            show_images_arg,
            gazebo_tran_arg,
            dl_arg,
            yolo_arg,
            gazebo_server,
            delayed_visualization,
        ]
    )
