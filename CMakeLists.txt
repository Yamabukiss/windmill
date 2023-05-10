cmake_minimum_required(VERSION 3.0.2)
project(windmill)


find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        nodelet
        pluginlib
        tf
        rm_msgs
        roslib
        )

generate_dynamic_reconfigure_options(
        cfg/dynamic.cfg
)

find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)


catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp dynamic_reconfigure std_msgs cv_bridge dynamic_reconfigure rm_msgs
        LIBRARIES ${PROJECT_NAME}
)

add_executable(${PROJECT_NAME} src/functions.cpp src/windmill.cpp)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
        )
