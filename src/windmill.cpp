#include "windmill/windmill.h"

void Windmill::drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes) {
//    cv::Mat image = bgr.clone();
    std::string label_array[2]={"Red","Blue"};
    static int src_w = bgr.cols;
    static int src_h = bgr.rows;
    static float width_ratio = (float)src_w / (float)image_size_;
    static float height_ratio = (float)src_h / (float)image_size_;
    if (!bboxes.empty())
    {
        for (size_t i = 0; i < bboxes.size(); i++) {
            const BoxInfo &bbox = bboxes[i];
//            if (bbox.label==1 || bbox.label==3) continue;
            cv::Point2f center ((bbox.x1+bbox.x2+bbox.x3+bbox.x4)/4*width_ratio,(bbox.y1+bbox.y2+bbox.y3+bbox.y4)/4*height_ratio);

            std::vector<cv::Point2f> points_vec;
            points_vec.emplace_back(center);
            points_vec.emplace_back(cv::Point2f(bbox.x1* width_ratio,bbox.y1* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x2* width_ratio,bbox.y2* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x3* width_ratio,bbox.y3* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x4* width_ratio,bbox.y4* height_ratio));

//            getPnP(points_vec,bbox.label);
            static cv::Scalar color = cv::Scalar(205,235,255);
            cv::line(bgr,points_vec[1],points_vec[2],color,1);
            cv::line(bgr,points_vec[2],points_vec[3],color,1);
            cv::line(bgr,points_vec[3],points_vec[4],color,1);
            cv::line(bgr,points_vec[4],points_vec[1],color,1);
            cv::circle(bgr,points_vec[0],3,color,2);
            cv::putText(bgr,label_array[bbox.label],cv::Point2f(std::max(float(0),points_vec[1].x-10),std::max(points_vec[1].y-10,float(0))),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,1,color,2);
//            cv::putText(bgr,std::to_string(bbox.score),cv::Point2f(std::max(float(0),points_vec[1].x-10),std::max(points_vec[1].y-10,float(0))),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,1,color,2);
        }
    }

}

inline void Windmill::modelProcess(const cv::Mat& image)
{
    cv::Mat resized_img;
    resizeUniform(image, resized_img, cv::Size(image_size_, image_size_));

//    auto results = detect(resized_img, score_thresh_, nms_thresh_);
    detect(resized_img, score_thresh_);

}

void Windmill::cvProcess(const cv::Mat& image)
{
    cv::Mat gray, threshold;
    cv::cvtColor(image,gray,CV_BGR2GRAY);
    cv::threshold(gray,threshold,threshold_,255,CV_THRESH_BINARY);
    binary_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(),"mono8" , threshold).toImageMsg());

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(threshold,contours,cv::RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

    if (!hull_vec_.empty())
        hull_vec_.clear();

    for (auto& contour : contours)
    {
        std::vector<cv::Point> hull;
        cv::convexHull(contour,hull, true);
        bool area_judge = cv::contourArea(hull) >= min_area_threshold_ && cv::contourArea(hull) < max_area_threshold_;
        if (cv::matchShapes(contour,r_contour_,cv::CONTOURS_MATCH_I2,0) <= hull_bias_ && area_judge)
        {
//            cv::polylines(cv_image_->image,hull, true,cv::Scalar(0,255,0),2);
            hull_vec_.push_back(hull);
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
    int32_t r_array[2];

    if (!hull_vec_.empty())
    {
        std::sort(hull_vec_.begin(), hull_vec_.end(), [&](const auto &v1, const auto &v2){return cv::matchShapes(v1,r_contour_,cv::CONTOURS_MATCH_I2,0) < cv::matchShapes(v2,r_contour_,cv::CONTOURS_MATCH_I2,0);});
        cv::polylines(cv_image_->image,hull_vec_[0], true,cv::Scalar(0,255,0),2);
        auto moment = cv::moments(hull_vec_[0]);
        int cx = int(moment.m10 / moment.m00);
        int cy = int(moment.m01/  moment.m00);
//        data.data.emplace_back(cx);
//        data.data.emplace_back(cy);
        r_array[0] = static_cast<int32_t>(cx) * 1440 / image_size_;
        r_array[1] = static_cast<int32_t>(cy) * 1080 / image_size_;

        if (!box_result_vec_.empty())
        {
            static float width_ratio = (float)cv_image_->image.cols / (float)image_size_;
            static float height_ratio = (float)cv_image_->image.rows / (float)image_size_;

            auto point = kalman_filter_ptr_->getAngle(cx, cy, box_result_vec_, width_ratio, height_ratio);
            cv::circle(cv_image_->image,point,8,cv::Scalar (255,255,0),cv::FILLED);

            if (kalman_filter_ptr_->object_loss_)
                kalman_filter_ptr_->object_loss_ = false;

            poly_array[0] = static_cast<int32_t>(box_result_vec_[0].x1) * 1440 / image_size_;
            poly_array[1] = static_cast<int32_t>(box_result_vec_[0].y1) * 1080 / image_size_;
            poly_array[2] = static_cast<int32_t>(box_result_vec_[0].x4) * 1440 / image_size_;
            poly_array[3] = static_cast<int32_t>(box_result_vec_[0].y4) * 1080 / image_size_;
            poly_array[4] = static_cast<int32_t>(box_result_vec_[0].x3) * 1440 / image_size_;
            poly_array[5] = static_cast<int32_t>(box_result_vec_[0].y3) * 1080 / image_size_;
            poly_array[6] = static_cast<int32_t>(box_result_vec_[0].x2) * 1440 / image_size_;
            poly_array[7] = static_cast<int32_t>(box_result_vec_[0].y2) * 1080 / image_size_;
        }

        else
            kalman_filter_ptr_->object_loss_ = true;
    }
    else
        kalman_filter_ptr_->object_loss_ = true;

    data.id = 10;

    memcpy(&data.pose.orientation.x, &poly_array[0],sizeof (int32_t) * 2);
    memcpy(&data.pose.orientation.y, &poly_array[2],sizeof (int32_t) * 2);
    memcpy(&data.pose.orientation.z, &poly_array[4],sizeof (int32_t) * 2);
    memcpy(&data.pose.orientation.w, &poly_array[6],sizeof (int32_t) * 2);

    memcpy(&data.pose.position.x, &r_array[0],sizeof (int32_t));
    memcpy(&data.pose.position.y, &r_array[1],sizeof (int32_t));



    data_array.detections.emplace_back(data);
    data_array.header.stamp = cv_image_->header.stamp;

    point_publisher_.publish(data_array);
}

//void Windmill::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
void Windmill::receiveFromCam(const sensor_msgs::ImageConstPtr& msg)
{
    cur_time_stamp_ = ros::Time::now().toSec();
    if (!windmill_work_signal_)
        return;
//    cv_image_ = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::RGB8);
    cv_image_ = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::RGB8);
//    cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(msg, msg->encoding));
    threading();
    result_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(),cv_image_->encoding , cv_image_->image).toImageMsg());
    prev_time_stamp_ = cur_time_stamp_;
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "windmill");

    std::cout << "start init model" << std::endl;
    Windmill detect;
    std::cout << "success" << std::endl;
    detect.kalman_filter_ptr_ = new Kalman();
    detect.onInit();

    while (ros::ok())
    {
        ros::spinOnce();
    }
}
