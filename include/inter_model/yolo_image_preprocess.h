#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "common.h"
#include "image_utils.h"
#include "model_params.hpp"
#include "yolov8.h"

class ImagePreProcess {
    // padding resize
 public:
    ImagePreProcess(int width, int height, int target_size);   // 只传一个值，默认输入图片是正方形的构造函数
    ImagePreProcess(int width, int height, int target_width, int target_height);
    std::unique_ptr<cv::Mat> Convert(const cv::Mat &src);
    const letterbox_t &get_letter_box() { return letterbox_; };

 private:
    double scale_;            // 最大边放缩比例
    int padding_x_;           // 宽度上填充大小
    int padding_y_;           // 高度上填充大小
    cv::Size new_size_; 
    // int target_size_;         // 只传一个值的边长
    int target_width_;        // 传了两个值的宽度
    int target_height_;       // 传了两个值的高度
    letterbox_t letterbox_;   // pad_w, pad_h, pad_r
};

// 裁剪结构体
struct CropInfo {
    cv::Mat cropped_img;
    int x_start;
    int y_start;
};

// 保存裁剪的目标检测结果
struct DetectionResult {
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> classes;
};


// 将XYWH转换为AABB
image_rect_t convertObbToAabb(const image_obb_box_t& obb);
// 从 input.data 获取图片的 cv::Mat 格式
cv::Mat convertDataToCvMat(const det_model_input& input);

// 将图片裁剪成一组正方形块
std::vector<CropInfo> cropImage(const cv::Mat &image);

// 偏移检测框坐标
std::vector<cv::Rect> offsetBboxes(const std::vector<cv::Rect> &bboxes, int x_offset, int y_offset);

// 计算两个检测框的 IoU
float calculateIoU(const image_rect_t &box1, const image_rect_t &box2);

// 多个裁剪的nms
std::vector<int> crop_nms_with_classes(const std::vector<image_rect_t> &bboxes, const std::vector<float> &confidences, 
                                       const std::vector<int> &classes, float iou_threshold);
std::vector<int> crop_nms_with_classes(const std::vector<image_obb_box_t>& bboxes, const std::vector<float>& confidences,
    const std::vector<int>& classes, float iou_threshold);
// 正方形检测的处理完整流程
int processDetBySquare(rknn_app_context_t *app_ctx, object_detect_result_list *od_results, det_model_input input_data,
                        int input_width, int input_height, float nms_thresh, float box_thresh, bool enable_logger);
int processDetBySquareObb(rknn_app_context_t* app_ctx, object_detect_obb_result_list* od_results, det_model_input input_data,
    int input_width, int input_height, float nms_thresh, float box_thresh, bool enable_logger);