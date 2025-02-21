/* 分类模型的图像处理 */

#include "outer_model/model_params.hpp"
#include "common.h"
#include "opencv2/opencv.hpp"

std::unique_ptr<cv::Mat> Resize(const cv::Mat& src, int target_width, int target_height);
std::unique_ptr<cv::Mat> Resize(const cv::Mat& src, int orignal_width, int orignal_height, int target_width, int target_height);
cv::Mat convertToCvMat(const det_model_input& input);