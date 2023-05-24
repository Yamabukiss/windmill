#include "windmill/windmill.h"

void Windmill::drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes) {
    std::string label_array[2]={"Red","Blue"};
    static int src_w = bgr.cols;
    static int src_h = bgr.rows;
    static float width_ratio = (float)src_w / (float)image_size_;
    static float height_ratio = (float)src_h / (float)image_size_;
    if (!bboxes.empty())
    {
        for (size_t i = 0; i < bboxes.size(); i++) {
            const BoxInfo &bbox = bboxes[i];
            cv::Point2f center ((bbox.x1+bbox.x2+bbox.x3+bbox.x4)/4*width_ratio,(bbox.y1+bbox.y2+bbox.y3+bbox.y4)/4*height_ratio);

            std::vector<cv::Point2f> points_vec;
            points_vec.emplace_back(center);
            points_vec.emplace_back(cv::Point2f(bbox.x1* width_ratio,bbox.y1* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x2* width_ratio,bbox.y2* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x3* width_ratio,bbox.y3* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x4* width_ratio,bbox.y4* height_ratio));

            static cv::Scalar color = cv::Scalar(205,235,255);
            cv::line(bgr,points_vec[1],points_vec[2],color,1);
            cv::line(bgr,points_vec[2],points_vec[3],color,1);
            cv::line(bgr,points_vec[3],points_vec[4],color,1);
            cv::line(bgr,points_vec[4],points_vec[1],color,1);
            cv::circle(bgr,points_vec[0],3,color,cv::FILLED);
//                        cv::putText(bgr,label_array[bbox.label],cv::Point2f(std::max(float(0),points_vec[1].x-10),std::max(points_vec[1].y-10,float(0))),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,1,color,2);
            //            cv::putText(bgr,std::to_string(bbox.score),cv::Point2f(std::max(float(0),points_vec[1].x-10),std::max(points_vec[1].y-10,float(0))),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,1,color,2);
        }
    }

}

inline void Windmill::modelProcess(const cv::Mat& image) {
  cv::Mat tmp_image = image.clone();
  detect(tmp_image, score_thresh_);
}

void Windmill::cvProcess(const cv::Mat& image)
{
    cv::Mat hsv, threshold;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    if (red_)
        cv::inRange(hsv,cv::Scalar(red_lower_hsv_h_,red_lower_hsv_s_,red_lower_hsv_v_),cv::Scalar(red_upper_hsv_h_,red_upper_hsv_s_,red_upper_hsv_v_),threshold);
    else
        cv::inRange(hsv,cv::Scalar(blue_lower_hsv_h_,blue_lower_hsv_s_,blue_lower_hsv_v_),cv::Scalar(blue_upper_hsv_h_,blue_upper_hsv_s_,blue_upper_hsv_v_),threshold);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + 2 * morph_size_, 1 + 2 * morph_size_),
                                               cv::Point(-1, -1));
    cv::morphologyEx(threshold, threshold, morph_type_, kernel, cv::Point(-1, -1), morph_iterations_);

    binary_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(),"mono8" , threshold).toImageMsg());

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(threshold,contours,cv::RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

    if (!hull_vec_.empty())
        hull_vec_.clear();

    for (auto& contour : contours)
    {
      bool area_judge = cv::contourArea(contour) >= min_area_threshold_ &&
                        cv::contourArea(contour) < max_area_threshold_;
      if (cv::matchShapes(contour,r_contour_,cv::CONTOURS_MATCH_I2,0) <= hull_bias_ && area_judge) {
        auto rect = cv::boundingRect(contour);
        rect += cv::Point(0.25 * rect.width, 0.25 * rect.height);
        rect -= cv::Size(0.25 * rect.width, 0.25 * rect.height);
        auto mask = threshold(rect);
        if (static_cast<double>(cv::countNonZero(mask)) / rect.area() >=
            area_duty_) {
          hull_vec_.push_back(contour);
        }
      }
    }

}

void Windmill::threading()
{
    std::thread thread_1(std::bind(&Windmill::modelProcess, this, std::ref(cv_image_->image)));
    std::thread thread_2(std::bind(&Windmill::cvProcess, this, std::ref(cv_image_->image)));
    thread_1.join();
    if (thread_2.joinable())
        thread_2.join();
//    modelProcess(cv_image_->image);
    drawBboxes(cv_image_->image, box_result_vec_);

    rm_msgs::TargetDetectionArray data_array;
    rm_msgs::TargetDetection data;

    int32_t poly_array[8];
    float r_array[2];
    float center_array[2];

    if (!hull_vec_.empty() && !box_result_vec_.empty())
    {
        std::vector<std::pair<cv::Point, double>> radian_vec;
        radian_vec.reserve(hull_vec_.size());
        auto p1 = cv::Point2f(box_result_vec_[0].x1, box_result_vec_[0].y1);
        auto p2 = cv::Point2f(box_result_vec_[0].x2, box_result_vec_[0].y2);
        auto p3 = cv::Point2f(box_result_vec_[0].x3, box_result_vec_[0].y3);
        auto p4 = cv::Point2f(box_result_vec_[0].x4, box_result_vec_[0].y4);
        auto center = (p1 + p2 + p3 + p4) / 4;
        center_array[0] = center.x * 1440 / image_size_;
        center_array[1] = center.y * 1080 / image_size_;
        for (const auto &contour : hull_vec_)
        {
            auto moment = cv::moments(contour);
            int cx = int(moment.m10 / moment.m00);
            int cy = int(moment.m01/  moment.m00);
            auto r = cv::Point2f(cx, cy);
            int p34_2_p12_vec_x = (p3.x + p4.x) / 2 - (p1.x + p2.x) / 2;
            int p34_2_p12_vec_y = (p3.y + p4.y) / 2 - (p1.y + p2.y) / 2;
//
            int r_2_p12_vec_x = r.x - (p1.x + p2.x) / 2;
            int r_2_p12_vec_y = r.y - (p1.y + p2.y) / 2;
            double cos_theta = (p34_2_p12_vec_x * r_2_p12_vec_x + p34_2_p12_vec_y * r_2_p12_vec_y) /
                    (sqrt(pow(p34_2_p12_vec_x, 2) + pow(p34_2_p12_vec_y, 2)) * sqrt(pow(r_2_p12_vec_x, 2) + pow(r_2_p12_vec_y, 2)));
            double radian = acos(cos_theta);
            radian_vec.emplace_back( std::make_pair(r, radian));
        }

        std::sort(radian_vec.begin(), radian_vec.end(), [](const auto &v1, const auto &v2){ return v1.second < v2.second; });
        auto r = radian_vec[0].first;
        cv::circle(cv_image_->image, r, 3,
                   cv::Scalar(0, 255, 0), cv::FILLED);
        r_array[0] = r.x * 1440 / image_size_;
        r_array[1] = r.y * 1080 / image_size_;
        poly_array[0] = static_cast<int32_t>(box_result_vec_[0].x1) * 1440 / image_size_;
        poly_array[1] = static_cast<int32_t>(box_result_vec_[0].y1) * 1080 / image_size_;
        poly_array[2] = static_cast<int32_t>(box_result_vec_[0].x4) * 1440 / image_size_;
        poly_array[3] = static_cast<int32_t>(box_result_vec_[0].y4) * 1080 / image_size_;
        poly_array[4] = static_cast<int32_t>(box_result_vec_[0].x3) * 1440 / image_size_;
        poly_array[5] = static_cast<int32_t>(box_result_vec_[0].y3) * 1080 / image_size_;
        poly_array[6] = static_cast<int32_t>(box_result_vec_[0].x2) * 1440 / image_size_;
        poly_array[7] = static_cast<int32_t>(box_result_vec_[0].y2) * 1080 / image_size_;

        data.id = 10;
        memcpy(&data.pose.position.x, &r_array[0],sizeof (float) * 2);
        memcpy(&data.pose.position.y, &center_array[0],sizeof (float) * 2);
        data.pose.position.z = 1;
        memcpy(&data.pose.orientation.x, &poly_array[0],sizeof (int32_t) * 2);
        memcpy(&data.pose.orientation.y, &poly_array[2],sizeof (int32_t) * 2);
        memcpy(&data.pose.orientation.z, &poly_array[4],sizeof (int32_t) * 2);
        memcpy(&data.pose.orientation.w, &poly_array[6],sizeof (int32_t) * 2);

        data_array.detections.emplace_back(data);

    }


    data_array.header.stamp = cv_image_->header.stamp;

    point_publisher_.publish(data_array);
}

void Windmill::receiveFromCam(const sensor_msgs::ImageConstPtr& msg)
{
    if (!windmill_work_signal_)
        return;
    cv_image_ = cv_bridge::toCvCopy(msg, "bgr8");
//    cv_image_ = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::RGB8);
    cv::resize(cv_image_->image, cv_image_->image, cv::Size(image_size_,image_size_));
    threading();
    result_publisher_.publish(
//        cv_bridge::CvImage(std_msgs::Header(), "rgb8", cv_image_->image)
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_->image)
            .toImageMsg());
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "windmill");

    std::cout << "start init model" << std::endl;
    Windmill detect;
    std::cout << "success" << std::endl;
    detect.onInit();

    while (ros::ok())
    {
        ros::spinOnce();
    }
}
