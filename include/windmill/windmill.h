#pragma once

#include <inference_engine.hpp>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <windmill/dynamicConfig.h>
#include <sensor_msgs/CompressedImage.h>
#include <thread>
#include <rm_msgs/TargetDetection.h>
#include <rm_msgs/TargetDetectionArray.h>
#include <ros/package.h>
//#include "windmill/kalmanFilter.h"
#include "nodelet/nodelet.h"
#include <pluginlib/class_list_macros.h>


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
namespace windmill {

    class Windmill : public nodelet::Nodelet {
    public:
        Windmill() : windmill_work_signal_(true) {};

        virtual ~Windmill() = default;

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

        void detect(cv::Mat image, double score_threshold);

        void onInit() override;

        void modelProcess(const cv::Mat &image);

        void cvProcess(const cv::Mat &image);

        void threading();

        void receiveFromCam(const sensor_msgs::ImageConstPtr &image);


        void resizeUniform(const cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size);

        void drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes);

        void dynamicCallback(windmill::dynamicConfig &config);

        void preProcess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob);

        double getL2Distance(const cv::Point2f &p1, const cv::Point2f &p2);

        void decodeInfer(const float *&cls_pred, const float *&dis_pred, int stride,
                         double threshold,
                         std::vector<std::vector<BoxInfo>> &results);

        BoxInfo disPred2Bbox(const float *&box_det, int label, double score, int x,
                             int y, int stride);

        static void nms(std::vector<BoxInfo> &result);


        dynamic_reconfigure::Server<windmill::dynamicConfig> server_;
        dynamic_reconfigure::Server<windmill::dynamicConfig>::CallbackType callback_;
        double score_thresh_{};
        double hull_bias_{};
        int min_area_threshold_{};
        int max_area_threshold_{};
        bool windmill_work_signal_;
        int morph_type_;
        int morph_iterations_;
        int morph_size_;
        bool red_;
        int red_lower_hsv_h_;
        int red_lower_hsv_s_;
        int red_lower_hsv_v_;
        int red_upper_hsv_h_;
        int red_upper_hsv_s_;
        int red_upper_hsv_v_;

        int blue_lower_hsv_h_;
        int blue_lower_hsv_s_;
        int blue_lower_hsv_v_;
        int blue_upper_hsv_h_;
        int blue_upper_hsv_s_;
        int blue_upper_hsv_v_;
        double area_duty_;

        std::vector<cv::Point> r_contour_;
        std::vector<BoxInfo> box_result_vec_;
        std::vector<BoxInfo> prev_box_result_vec_;
        std::vector<std::vector<cv::Point>> hull_vec_;
        std::string input_name_;
        cv_bridge::CvImagePtr cv_image_;
        ros::NodeHandle nh_;
        ros::Subscriber img_subscriber_;
        ros::Publisher result_publisher_;
        ros::Publisher binary_publisher_;
        ros::Publisher point_publisher_;
        int num_class_ = 2;
        int image_size_ = 416;
        std::mutex mutex_;
//    Kalman * kalman_filter_ptr_{};
    };
}
