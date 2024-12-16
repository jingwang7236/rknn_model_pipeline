#ifndef MY_MODEL_H
#define MY_MODEL_H

#include "opencv2/opencv.hpp"
#include "outer_model/model_params.hpp"
#include "common.h"

int inference_retinanet_model(rknn_app_context_t *app_ctx, cv::Mat src_img, ssd_det_result *out_result, const int num_class);

int inference_retinaface_model(rknn_app_context_t *app_ctx, image_buffer_t *img, retinaface_result *out_result);

int inference_classify_model(rknn_app_context_t *app_ctx, cv::Mat src_img, rknn_output* outputs);


#endif // MY_MODEL_H