#include "windmill/windmill.h"

void Windmill::onInit()
{
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork("/home/yamabuki/detect_ws/src/windmill/picodet_sim.xml");
    // prepare input settings
    InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
    input_name_ = inputs_map.begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
    // prepare output settings
    InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());

    for (auto &output_info : outputs_map) {
        output_info.second->setPrecision(InferenceEngine::Precision::FP32);
    }

    std::map<std::string, std::string> config = {
            { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO },
            { InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NUMA },
            { InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                    InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA },
//            { InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "16" },
    };


    // get network
    network_ = ie.LoadNetwork(model, "CPU",config);
    infer_request_ = network_.CreateInferRequest();

//    img_subscriber_= nh_.subscribe("/hk_camera/image_raw", 1, &Windmill::receiveFromCam,this);
    img_subscriber_= nh_.subscribe("/hk_camera/camera/image_raw/compressed", 1, &Windmill::receiveFromCam,this);
    result_publisher_ = nh_.advertise<sensor_msgs::Image>("result_publisher", 1);
//    direction_publisher_ = nh_.advertise<std_msgs::Int8>("direction_publisher", 1);
    pnp_publisher_ = nh_.advertise<geometry_msgs::Pose>("pnp_publisher", 1);
    flag_publisher_ = nh_.advertise<std_msgs::Int8>("flag_publisher", 1);

    callback_ = boost::bind(&Windmill::dynamicCallback, this, _1);
    server_.setCallback(callback_);
    object_points_.emplace_back(cv::Point3f(-0.135,-0.135,0));
    object_points_.emplace_back(cv::Point3f(0.135,-0.135,0));
    object_points_.emplace_back(cv::Point3f(0.135,0.135,0));
    object_points_.emplace_back(cv::Point3f(-0.135,0.135,0));
    distortion_coefficients_ =(cv::Mat_<double>(1,5)<<-0.066342, 0.081260, -0.000165, 0.000870, 0.000000);
    camera_matrix_=(cv::Mat_<double>(3,3)<<801.64373,    0.     ,  322.80906,
            0.     , 1068.42759,  318.9301 ,
            0.     ,    0.     ,    1.    );
    static tf2_ros::TransformListener tfListener(tf_buffer_);
}

Windmill::Windmill() {}

void Windmill::dynamicCallback(windmill::dynamicConfig &config)
{
    nms_thresh_=config.nms_thresh;
    score_thresh_=config.score_thresh;
    ROS_INFO("Seted Complete");
}


Windmill::~Windmill() {}

void Windmill::preProcess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob) {
    int img_w = image.cols;
    int img_h = image.rows;
    int channels = 3;

    InferenceEngine::MemoryBlob::Ptr mblob =
            InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        THROW_IE_EXCEPTION
                << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    auto mblobHolder = mblob->wmap();
    float *blob_data = mblobHolder.as<float *>();

    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < img_h; h++) {
            for (size_t w = 0; w < img_w; w++) {
                blob_data[c * img_w * img_h + h * img_w + w] =
                        (float)image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

void Windmill::getPnP(const std::vector<cv::Point2f> &added_weights_points,int label)
{
    if(label==2 || label==3)
    {
        static std::vector<cv::Point2f> image_points;
        image_points.emplace_back(added_weights_points[1]);
        image_points.emplace_back(added_weights_points[2]);
        image_points.emplace_back(added_weights_points[3]);
        image_points.emplace_back(added_weights_points[4]);
        cv::solvePnP(object_points_,image_points,camera_matrix_,distortion_coefficients_,rvec_,tvec_);
        cv::Mat r_mat = cv::Mat_<double>(3, 3);

        cv::Rodrigues(rvec_, r_mat);
        tf::Matrix3x3 tf_rotate_matrix(r_mat.at<double>(0, 0), r_mat.at<double>(0, 1), r_mat.at<double>(0, 2),
                                       r_mat.at<double>(1, 0), r_mat.at<double>(1, 1), r_mat.at<double>(1, 2),
                                       r_mat.at<double>(2, 0), r_mat.at<double>(2, 1), r_mat.at<double>(2, 2));

        tf::Quaternion quaternion;
        double r;
        double p;
        double y;

        static geometry_msgs::Pose  pose;

        pose.position.x=tvec_.at<double>(0,0);
        pose.position.y=tvec_.at<double>(0,1);
        pose.position.z=tvec_.at<double>(0,2);

        tf_rotate_matrix.getRPY(r, p, y);
        quaternion.setRPY(r,p,y);
        pose.orientation.x=quaternion.x();
        pose.orientation.y=quaternion.y();
        pose.orientation.z=quaternion.z();
        pose.orientation.w=quaternion.w();

        pnp_publisher_.publish(pose);

        geometry_msgs::TransformStamped pose_in , pose_out;

        tf2::Quaternion tf_quaternion;
        tf_quaternion.setRPY(y,p,r);
        geometry_msgs::Quaternion quat_msg = tf2::toMsg(tf_quaternion);

        pose_in.transform.rotation.x = quat_msg.x;
        pose_in.transform.rotation.y = quat_msg.y;
        pose_in.transform.rotation.z = quat_msg.z;
        pose_in.transform.rotation.w = quat_msg.w;

        tf2::doTransform(pose_in, pose_out, tf_buffer_.lookupTransform("camera_optical_frame", "base_link", ros::Time(0)));

        pose_out.transform.translation.x = pose.position.x;
        pose_out.transform.translation.y = pose.position.y;
        pose_out.transform.translation.z = pose.position.z;

        tf::Transform transform;

        transform.setOrigin(tf::Vector3(pose_out.transform.translation.x, pose_out.transform.translation.y,
                                        pose_out.transform.translation.z));
        transform.setRotation(tf::Quaternion(pose_out.transform.rotation.x, pose_out.transform.rotation.y,
                                             pose_out.transform.rotation.z, pose_out.transform.rotation.w));
//        transform.setRotation(tf::Quaternion(pose.orientation.x, pose.orientation.y,
//                                             pose.orientation.z, pose.orientation.w));

        tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_optical_frame", "exchanger"));
//        flag_.data = 1;
//        flag_publisher_.publish(flag_);
        image_points.clear();
    }
    else
    {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(2,2,2));
        transform.setRotation(tf::Quaternion(0,0,0,1));
        tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_optical_frame", "exchanger"));
//        flag_.data = 0;
//        flag_publisher_.publish(flag_);

    }
}


std::vector<BoxInfo> Windmill::detect(cv::Mat image, double score_threshold,
                                     double nms_threshold) {
    InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);

    preProcess(image, input_blob);

    // do inference
    infer_request_.Infer();
//    infer_request_.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    // get output
    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class_);

    for (const auto &head_info : this->heads_info_) {
        const InferenceEngine::Blob::Ptr dis_pred_blob =
                infer_request_.GetBlob(head_info.dis_layer);
        const InferenceEngine::Blob::Ptr cls_pred_blob =
                infer_request_.GetBlob(head_info.cls_layer);

        auto mdis_pred =
                InferenceEngine::as<InferenceEngine::MemoryBlob>(dis_pred_blob);
        auto mdis_pred_holder = mdis_pred->rmap();
        const float *dis_pred = mdis_pred_holder.as<const float *>();

        auto mcls_pred =
                InferenceEngine::as<InferenceEngine::MemoryBlob>(cls_pred_blob);
        auto mcls_pred_holder = mcls_pred->rmap();
        const float *cls_pred = mcls_pred_holder.as<const float *>();
        this->decodeInfer(cls_pred, dis_pred, head_info.stride, score_threshold,
                          results);
    }

    std::vector<BoxInfo> dets;
    for (int i = 0; i < (int)results.size(); i++) {
        this->nms(results[i], nms_threshold);

        for (auto &box : results[i]) {
            dets.push_back(box);
        }
    }
    return dets;
}

void Windmill::decodeInfer(const float *&cls_pred, const float *&dis_pred,
                          int stride, double threshold,
                          std::vector<std::vector<BoxInfo>> &results) {
    int feature_h = ceil((float)image_size_ / stride);
    int feature_w = ceil((float)image_size_ / stride);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class_; label++) {
            if (cls_pred[idx * num_class_ + label] > score) {
                score = cls_pred[idx * num_class_ + label];
                cur_label = label;
            }
        }
        if (score > threshold)
        {
            const float *bbox_pred = dis_pred + idx  * 8;
            results[cur_label].push_back(
                    this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
        }

    }
}

void Windmill::resizeUniform(cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size){
    if (src.cols==dst_size.width && src.rows==dst_size.height) dst = src.clone();
    else
    {
        int dst_w = dst_size.width;
        int dst_h = dst_size.height;
//        dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));
        cv::resize(src,dst,cv::Size(dst_w,dst_h));
    }
}


BoxInfo Windmill::disPred2Bbox(const float *&box_det, int label, double score,
                              int x, int y, int stride) {
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
//    float ct_x = (x + 0.5) * stride;
//    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(8);
    for (int i = 0; i < 8; i++) {
        float dis = box_det[i];
        dis *= stride;
        dis_pred[i] = dis;
    }
    float x1 = (std::max)(ct_x + dis_pred[0], .0f);
    float y1 = (std::max)(ct_y + dis_pred[1], .0f);
    float x2 = (std::min)(ct_x + dis_pred[2], (float)this->image_size_);
    float y2 = (std::max)(ct_y + dis_pred[3], .0f);

    float x3 = (std::min)(ct_x + dis_pred[4], (float)this->image_size_);
    float y3 = (std::min)(ct_y + dis_pred[5], (float)this->image_size_);
    float x4 = (std::max)(ct_x + dis_pred[6], .0f);
    float y4 = (std::min)(ct_y + dis_pred[7], (float)this->image_size_);
    return BoxInfo{x1 , y1 , x2 , y2 , x3 , y3 , x4 , y4 , score , label };
}

void Windmill::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {

    std::sort(input_boxes.begin(), input_boxes.end(),
              [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    if (input_boxes.size() > 1)
        input_boxes.erase(input_boxes.begin()+1,input_boxes.end());
//    std::vector<float> vArea(input_boxes.size());
//    for (int i = 0; i < int(input_boxes.size()); ++i) {
//        vArea[i] = (input_boxes.at(i).x3 - input_boxes.at(i).x1 + 1) *
//                   (input_boxes.at(i).y3 - input_boxes.at(i).y1 + 1);
//    }
//    for (int i = 0; i < int(input_boxes.size()); ++i) {
//        for (int j = i + 1; j < int(input_boxes.size());) {
//            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
//            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
//            float xx2 = (std::min)(input_boxes[i].x3, input_boxes[j].x3);
//            float yy2 = (std::min)(input_boxes[i].y3, input_boxes[j].y3);
//            float w = (std::max)(float(0), xx2 - xx1 + 1);
//            float h = (std::max)(float(0), yy2 - yy1 + 1);
//            float inter = w * h;
//            float ovr = inter / (vArea[i] + vArea[j] - inter);
//            if (ovr >= NMS_THRESH) {
//                input_boxes.erase(input_boxes.begin() + j);
//                vArea.erase(vArea.begin() + j);
//            } else {
//                j++;
//            }
//        }
//    }
}
