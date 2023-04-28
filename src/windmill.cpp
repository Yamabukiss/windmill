#include "windmill/windmill.h"
#include <string>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

int g_num = 0;
void GetFileNames(std::string path,std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}



void Windmill::drawBboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes) {
    cv::Mat image = bgr.clone();
    std::string label_array[1]={"Red"};
    static int src_w = image.cols;
    static int src_h = image.rows;
    static float width_ratio = (float)src_w / (float)image_size_;
    static float height_ratio = (float)src_h / (float)image_size_;
    if (!bboxes.empty())
    {
        for (size_t i = 0; i < bboxes.size(); i++) {
            const BoxInfo &bbox = bboxes[i];
//            if (bbox.label==1) continue; // free the solid which is useless
            cv::Point2f center ((bbox.x1+bbox.x2+bbox.x3+bbox.x4)/4*width_ratio,(bbox.y1+bbox.y2+bbox.y3+bbox.y4)/4*height_ratio);

            std::vector<cv::Point2f> points_vec;
            points_vec.emplace_back(center);
            points_vec.emplace_back(cv::Point2f(bbox.x1* width_ratio,bbox.y1* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x2* width_ratio,bbox.y2* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x3* width_ratio,bbox.y3* height_ratio));
            points_vec.emplace_back(cv::Point2f(bbox.x4* width_ratio,bbox.y4* height_ratio));

//            getPnP(points_vec,bbox.label);
            static cv::Scalar color = cv::Scalar(205,235,255);
            cv::line(image,points_vec[1],points_vec[2],color,1);
            cv::line(image,points_vec[2],points_vec[3],color,1);
            cv::line(image,points_vec[3],points_vec[4],color,1);
            cv::line(image,points_vec[4],points_vec[1],color,1);
            cv::circle(image,points_vec[0],3,color,3);
            cv::putText(image,label_array[bbox.label],cv::Point2f(std::max(float(0),points_vec[1].x-10),std::max(points_vec[1].y-10,float(0))),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,1,color,2);
        }
//        std::cout<<"fick"<<std::endl;
    }

    cv::imwrite("/home/yamabuki/10GDisk/result_images/"+std::to_string(g_num)+".jpg",image);
    g_num++;

//    result_publisher_.publish(cv_bridge::CvImage(std_msgs::Header(),cv_image_->encoding , image).toImageMsg());

}

void Windmill::receiveFromCam(const sensor_msgs::ImageConstPtr& image)
{
    cv_image_ = boost::make_shared<cv_bridge::CvImage>(*cv_bridge::toCvShare(image, image->encoding));
    cv::Mat resized_img;

    resizeUniform(cv_image_->image, resized_img, cv::Size(image_size_, image_size_));

    auto results = detect(resized_img, score_thresh_, nms_thresh_);
    drawBboxes(cv_image_->image, results);
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "windmill");

    std::cout << "start init model" << std::endl;
    Windmill detect;
    std::cout << "success" << std::endl;
    detect.onInit();
    std::vector<std::string> vec;
    GetFileNames("/home/yamabuki/10GDisk/result_images",vec);
    for (auto &name : vec)
    {
        cv::Mat img = cv::imread(name);
        cv::Mat resized_img;
        detect.resizeUniform(img, resized_img, cv::Size(640, 640));
        auto results = detect.detect(resized_img, 0.2, 0.1);
        detect.drawBboxes(img, results);
    }

//    while (ros::ok())
//    {
//        ros::spinOnce();
//    }
}
