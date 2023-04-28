#pragma once

#include <inference_engine.hpp>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int8.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <windmill/dynamicConfig.h>
#include <std_msgs/Int8.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

typedef struct HeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    double score;
    int label;
} BoxInfo;

class Windmill {
public:
    Windmill();

    ~Windmill();

    InferenceEngine::ExecutableNetwork network_;
    InferenceEngine::InferRequest infer_request_;
    // static bool hasGPU;

    std::vector<HeadInfo> heads_info_{
            // cls_pred|dis_pred|stride
            {"transpose_0.tmp_0", "transpose_1.tmp_0", 8},
            {"transpose_2.tmp_0", "transpose_3.tmp_0", 16},
            {"transpose_4.tmp_0", "transpose_5.tmp_0", 32},
            {"transpose_6.tmp_0", "transpose_7.tmp_0", 64},
    };

    std::vector<BoxInfo> detect(cv::Mat image, double score_threshold,
                                double nms_threshold);

    void onInit();

    void receiveFromCam(const sensor_msgs::ImageConstPtr &image);

    void resizeUniform(cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size);

    void drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes);

    void dynamicCallback(windmill::dynamicConfig &config);

    void preProcess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob);

    void decodeInfer(const float *&cls_pred, const float *&dis_pred, int stride,
                     double threshold,
                     std::vector<std::vector<BoxInfo>> &results);

    BoxInfo disPred2Bbox(const float *&box_det, int label, double score, int x,
                         int y, int stride);

    static void nms(std::vector<BoxInfo> &result, float nms_threshold);


    void getPnP(const std::vector<cv::Point2f> &added_weights_points,int label);

    dynamic_reconfigure::Server<windmill::dynamicConfig> server_;
    dynamic_reconfigure::Server<windmill::dynamicConfig>::CallbackType callback_;
    double nms_thresh_;
    double score_thresh_;
    cv::Mat_<double> camera_matrix_;
    cv::Mat_<double> camera_matrix2_;
    cv::Mat_<double> distortion_coefficients_;
    cv::Mat_<double> rvec_;
    cv::Mat_<double> tvec_;
    cv::Mat_<double> rotate_mat_;
    std::vector<cv::Point3f> object_points_;
    std::string input_name_;
    cv_bridge::CvImagePtr cv_image_;
    ros::NodeHandle nh_;
    ros::Subscriber img_subscriber_;
    ros::Publisher result_publisher_;
    ros::Publisher pnp_publisher_;
    ros::Publisher flag_publisher_;
    tf2_ros::Buffer tf_buffer_;
    tf::TransformBroadcaster tf_broadcaster_;
    int num_class_ = 1;
    int image_size_ = 640;
};
