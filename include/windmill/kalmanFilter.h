#pragma once
#include "windmill/windmill.h"

 struct BoxInfo;

class Windmill;

class Kalman
{
public:

    Kalman()
    {
        cv::KalmanFilter kf(2,2,0);
        kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
        cv::setIdentity(kf.measurementMatrix);
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-5));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
        measurement_ = cv::Mat::zeros(2, 1, CV_32F);
        prev_errorcov_ = kf.errorCovPost.clone();
        kalman_filter_ = kf;
        object_loss_ = true;
    }
    ~Kalman() = default;

    cv::Point getAngle(int r_x, int r_y, const std::vector<BoxInfo> &box_result_vec, float width_ratio, float height_ratio);
    void resetKalmanFilter();
    bool distanceJudge(int cur_x, int cur_y, int predict_x, int predict_y);

    bool object_loss_;
    int process_noise_;
    int measurement_noise_;
    int distance_threshold_;
    int prev_mean_x_;
    int prev_mean_y_;
    double radian_scale_;
    double prev_radian_;
    double prev_delta2_t_;
    double cur_time_stamp_;
    double prev_time_stamp_;
    cv::Mat prev_errorcov_;
    cv::Mat measurement_;
    cv::KalmanFilter kalman_filter_;

};
