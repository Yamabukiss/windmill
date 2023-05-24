#include "windmill/windmill.h"
namespace windmill {
    void Windmill::onInit() {
        std::string model_path = "";
        ros::NodeHandle nh = getMTPrivateNodeHandle();
        nh.getParam("model_path", model_path);
        nh_ = ros::NodeHandle(nh, "windmill_node");
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork model = ie.ReadNetwork(model_path);
        // prepare input settings
        InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
        input_name_ = inputs_map.begin()->first;
        InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
        // prepare output settings
        InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());

        for (auto &output_info: outputs_map) {
            output_info.second->setPrecision(InferenceEngine::Precision::FP32);
        }
        std::map<std::string, std::string> config = {
                {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,      InferenceEngine::PluginConfigParams::NO},
                {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NUMA},
                {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                                                                           InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA},
//            { InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "16" },
        };


        // get network
        network_ = ie.LoadNetwork(model, "CPU", config);
        infer_request_ = network_.CreateInferRequest();

//    img_subscriber_= nh_.subscribe("/hk_camera/camera/image_raw", 1, &Windmill::receiveFromCam,this);
        img_subscriber_ = nh_.subscribe("/hk_camera/image_raw", 1, &Windmill::receiveFromCam, this);
//    img_subscriber_= nh_.subscribe("/image_rect", 1, &Windmill::receiveFromCam,this);
        result_publisher_ = nh_.advertise<sensor_msgs::Image>("/result_publisher", 1);
        binary_publisher_ = nh_.advertise<sensor_msgs::Image>("/binary_publisher", 1);
        point_publisher_ = nh_.advertise<rm_msgs::TargetDetectionArray>("/prediction", 1);
        callback_ = boost::bind(&Windmill::dynamicCallback, this, _1);
        server_.setCallback(callback_);


        cv::Mat temp_r = cv::imread(ros::package::getPath("windmill") + "/r.png");
        cv::resize(temp_r, temp_r, cv::Size(image_size_, image_size_));
        cv::Mat r;
        cv::cvtColor(temp_r, r, CV_BGR2GRAY);
        cv::Mat threshold_img;
        cv::threshold(r, threshold_img, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(threshold_img, contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        std::sort(contours.begin(), contours.end(),
                  [](const std::vector<cv::Point> &vec1, const std::vector<cv::Point> &vec2) {
                      return cv::contourArea(vec1) > cv::contourArea(vec2);
                  });

        r_contour_ = contours[0];
    }


    void Windmill::dynamicCallback(windmill::dynamicConfig &config) {
        red_ = config.red;
        red_lower_hsv_h_ = config.red_lower_hsv_h;
        red_lower_hsv_s_ = config.red_lower_hsv_s;
        red_lower_hsv_v_ = config.red_lower_hsv_v;
        red_upper_hsv_h_ = config.red_upper_hsv_h;
        red_upper_hsv_s_ = config.red_upper_hsv_s;
        red_upper_hsv_v_ = config.red_upper_hsv_v;

        blue_lower_hsv_h_ = config.blue_lower_hsv_h;
        blue_lower_hsv_s_ = config.blue_lower_hsv_s;
        blue_lower_hsv_v_ = config.blue_lower_hsv_v;
        blue_upper_hsv_h_ = config.blue_upper_hsv_h;
        blue_upper_hsv_s_ = config.blue_upper_hsv_s;
        blue_upper_hsv_v_ = config.blue_upper_hsv_v;

        morph_type_ = config.morph_type;
        morph_iterations_ = config.morph_iterations;
        morph_size_ = config.morph_size;


        score_thresh_ = config.score_thresh;
        hull_bias_ = config.hull_bias;
        min_area_threshold_ = config.min_area_threshold;
        max_area_threshold_ = config.max_area_threshold;
        area_duty_ = config.area_duty;

        ROS_INFO("Seted Complete");
    }

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
                            (float) image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }

    void Windmill::detect(cv::Mat image, double score_threshold) {
        InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);

        preProcess(image, input_blob);

        // do inference
        infer_request_.StartAsync();
        infer_request_.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

        // get output
        std::vector<std::vector<BoxInfo>> results;
        results.resize(this->num_class_);

        for (const auto &head_info: this->heads_info_) {
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

//    std::vector<BoxInfo> dets;
        if (!box_result_vec_.empty())
            box_result_vec_.clear();

        for (int i = 0; i < (int) results.size(); i++) {
            this->nms(results[i]);

            for (auto &box: results[i]) {
                box_result_vec_.push_back(box);
            }
        }
        std::sort(box_result_vec_.begin(), box_result_vec_.end(),
                  [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
        if (box_result_vec_.size() > 1)
            box_result_vec_.erase(box_result_vec_.begin() + 1, box_result_vec_.end());

    }

    void Windmill::decodeInfer(const float *&cls_pred, const float *&dis_pred,
                               int stride, double threshold,
                               std::vector<std::vector<BoxInfo>> &results) {
        int feature_h = ceil((float) image_size_ / stride);
        int feature_w = ceil((float) image_size_ / stride);
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
            if (score > threshold) {
                const float *bbox_pred = dis_pred + idx * 8;
                results[cur_label].push_back(
                        this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
            }

        }
    }

    void Windmill::resizeUniform(const cv::Mat &src, cv::Mat &dst, const cv::Size &dst_size) {
        if (src.cols == dst_size.width && src.rows == dst_size.height)
            dst = src.clone();
        else {
            int dst_w = dst_size.width;
            int dst_h = dst_size.height;
//        dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));
            cv::resize(src, dst, cv::Size(dst_w, dst_h));
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
        float x2 = (std::min)(ct_x + dis_pred[2], (float) this->image_size_);
        float y2 = (std::max)(ct_y + dis_pred[3], .0f);

        float x3 = (std::min)(ct_x + dis_pred[4], (float) this->image_size_);
        float y3 = (std::min)(ct_y + dis_pred[5], (float) this->image_size_);
        float x4 = (std::max)(ct_x + dis_pred[6], .0f);
        float y4 = (std::min)(ct_y + dis_pred[7], (float) this->image_size_);
        return BoxInfo{x1, y1, x2, y2, x3, y3, x4, y4, score, label};
    }

    void Windmill::nms(std::vector<BoxInfo> &input_boxes) {
//    for (auto it = input_boxes.begin(); it != input_boxes.end();)
//    {
//        if (it->label == 1 || it->label == 3)
//        {
//            it = input_boxes.erase(it);
//        }
//        else
//        {
//            ++it;
//        }
//    }

        std::sort(input_boxes.begin(), input_boxes.end(),
                  [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
        if (input_boxes.size() > 1)
            input_boxes.erase(input_boxes.begin() + 1, input_boxes.end());
    }

    double Windmill::getL2Distance(const cv::Point2f &p1, const cv::Point2f &p2) {
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));

    }
}
//PLUGINLIB_EXPORT_CLASS(windmill::Windmill, nodelet::Nodelet)