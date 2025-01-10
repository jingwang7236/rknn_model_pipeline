#pragma once

#include <string>
#include <memory>
#include <mutex>

#include "image_utils.h"
#include "common.h"
#include "rknn_api.h"

#include "model_params.hpp"

class Yolo {
  public:
    // 接受右值引用的构造函数
    Yolo(std::string &&model_path, YoloModelType model_type=YoloModelType::DETECTION, float nums_thresh=NMS_THRESH, float box_thresh=BOX_THRESH) : 
          model_path_(std::move(model_path)), model_type_(model_type), nms_thresh_(nums_thresh), box_thresh_(box_thresh) {}
    // 接受 const char* 的构造函数
    Yolo(const char *model_path, YoloModelType model_type=YoloModelType::DETECTION, float nums_thresh=NMS_THRESH, float box_thresh=BOX_THRESH) :
          model_path_(model_path), model_type_(model_type), nms_thresh_(nums_thresh), box_thresh_(box_thresh) {}
    ~Yolo();
    rknn_context *get_rknn_context();
    rknn_app_context_t *get_rknn_app_context_t();
    int init_model(rknn_context *ctx_in, bool copy_weight=false);
    int init_model(rknn_app_context_t *app_ctx, bool copy_weight=false);   // 兼容其他模型初始化方式（不推荐使用）
    int release_model();
    static int release_model(rknn_app_context_t *app_ctx);
    inline int get_model_width() const { return app_ctx_.model_width; };
    inline int get_model_height() const { return app_ctx_.model_height; };
    inline YoloModelType get_model_type() const { return model_type_; };
    inline float get_nms_thresh() const { return nms_thresh_; }
    inline float get_box_thresh() const { return box_thresh_; }
    inline void set_hyperparamter(int obj_class_num_ = 1, int kpt_num = 17, int result_num = 8400);
    inline void set_nms_thresh(float new_nms_thresh) { nms_thresh_ = new_nms_thresh; }
    inline void set_box_thresh(float new_box_thresh) { box_thresh_ = new_box_thresh; }
    inline bool get_enable_logger() const { return enable_logger_; };
    inline void set_enable_logger(bool new_enable_logger) { enable_logger_ = new_enable_logger; };
    // 推理函数
    int inference_yolo_model(void *image_buf, void *od_results, letterbox_t letter_box);

  private:
    rknn_app_context_t app_ctx_;
    rknn_context ctx_{0};
    std::string model_path_;      // 用于存储模型路径，无论输入是右值引用还是 const char*
    std::mutex outputs_lock_;
    YoloModelType model_type_;
    int obj_class_num_ = 1;
    int kpt_num_ = -1;
    int result_num_ = -1;
    float nms_thresh_;
    float box_thresh_;
    bool enable_logger_ = true;   // 默认打印日志
};
