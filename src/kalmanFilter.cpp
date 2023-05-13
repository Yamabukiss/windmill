#include "windmill/kalmanFilter.h"

void Kalman::resetKalmanFilter()
{
    kalman_filter_.statePost.at<float>(0) = prev_radian_ ;
    kalman_filter_.statePost.at<float>(1) = prev_radian_ ;
//    cv::setIdentity(kalman_filter_ptr_->kalman_filter_.errorCovPost, cv::Scalar::all(1));
}

bool Kalman::distanceJudge(int cur_x, int cur_y, int predict_x, int predict_y)
{
    double distance = sqrt(pow(cur_x - predict_x, 2) + pow(cur_y - predict_y, 2));

    if (distance > distance_threshold_)
        return true;
    else
        return false;
}

cv::Point Kalman::getAngle(int r_x, int r_y, const std::vector<BoxInfo> &box_result_vec, float width_ratio, float height_ratio)
{
    int mean_x = static_cast<int>( (box_result_vec[0].x1 + box_result_vec[0].x2 + box_result_vec[0].x3 + box_result_vec[0].x4 )/4 * width_ratio );
    int mean_y = static_cast<int>( (box_result_vec[0].y1 + box_result_vec[0].y2 + box_result_vec[0].y3 + box_result_vec[0].y4 )/4 * height_ratio ) ;

    if (object_loss_)
    {
        if (prev_radian_ > 0)
            resetKalmanFilter();
        prev_mean_x_ = mean_x;
        prev_mean_y_ = mean_y;
        return cv::Point (-1, -1);
    }
    else
    {
        int cur_vec_x = mean_x - r_x;
        int cur_vec_y = mean_y - r_y;

        int prev_vec_x = prev_mean_x_ - r_x;
        int prev_vec_y = prev_mean_y_ - r_y;
        double cos_theta = (cur_vec_x * prev_vec_x + cur_vec_y * prev_vec_y) / (sqrt(pow(cur_vec_x, 2) + pow(cur_vec_y, 2)) * sqrt(pow(prev_vec_x, 2) + pow(prev_vec_y, 2))) ;
        double radian = acos(cos_theta);

        kalman_filter_.statePost.at<float>(1) = radian;

        cv::Mat prediction = kalman_filter_.predict();

        measurement_.at<float>(0) = radian;
        kalman_filter_.correct(measurement_);

        double predict_radian = kalman_filter_.statePost.at<float>(0) * radian_scale_;
        int predict_vec_x = cos(predict_radian) * cur_vec_x  - sin(predict_radian) * cur_vec_y + r_x;
        int predict_vec_y = cos(predict_radian) * cur_vec_y  + sin(predict_radian) * cur_vec_x + r_y;

        if (distanceJudge(mean_x, mean_y, prev_mean_x_, prev_mean_y_) || predict_vec_x < 0 )
//        if (predict_vec_x < 0)
        {
            resetKalmanFilter();

            prediction = kalman_filter_.predict();

//            measurement_.at<float>(0) = radian;
//            kalman_filter_.correct(measurement_);
            predict_radian = prediction.at<float>(0) * radian_scale_;
            predict_vec_x = cos(predict_radian) * cur_vec_x  - sin(predict_radian) * cur_vec_y + r_x;
            predict_vec_y = cos(predict_radian) * cur_vec_y  + sin(predict_radian) * cur_vec_x + r_y;
        }
//        cv::circle(Windmill::cv_image_->image,cv::Point (predict_vec_x, predict_vec_y),8,cv::Scalar (255,0,255),2);

//        int new_vec_x = cos(radian) * cur_vec_x  - sin(radian) * cur_vec_y + r_x;
//        int new_vec_y = cos(radian) * cur_vec_y  + sin(radian) * cur_vec_x + r_y;
        prev_radian_ = radian;
        prev_mean_x_ = mean_x;
        prev_mean_y_ = mean_y;
        return cv::Point (predict_vec_x, predict_vec_y);
    }
}

