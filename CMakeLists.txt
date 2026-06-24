cmake_minimum_required(VERSION 3.8)
project(depth_image_to_pointcloud2)

# /data is commonly mounted from Windows in the test container, where creating
# install-time symlinks can be disallowed. Copy-install this package so the
# workspace can still be built with colcon's --symlink-install option.
set(AMENT_CMAKE_SYMLINK_INSTALL OFF CACHE BOOL "" FORCE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgproc)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(depth_image_to_pointcloud2
        src/main.cpp
        src/depth_image_to_pointcloud2_node.cpp
        src/classical_dehaze.cpp)
ament_target_dependencies(depth_image_to_pointcloud2
        rclcpp
        sensor_msgs
        cv_bridge)
target_include_directories(depth_image_to_pointcloud2 PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(depth_image_to_pointcloud2 ${OpenCV_LIBRARIES})

add_executable(yolo_node src/yolo_node.cpp)
ament_target_dependencies(yolo_node
        ament_index_cpp
        rclcpp
        sensor_msgs
        cv_bridge)
target_include_directories(yolo_node PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_node ${OpenCV_LIBRARIES})

install(TARGETS
        depth_image_to_pointcloud2
        yolo_node
        DESTINATION lib/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
        scripts/depth_image_to_pointcloud2_dl_node.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME depth_image_to_pointcloud2_dl)

install(DIRECTORY
        launch
        models
        rviz
        worlds
        yolo
        DESTINATION share/${PROJECT_NAME})

ament_package()
