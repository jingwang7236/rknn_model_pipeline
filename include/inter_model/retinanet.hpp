#ifndef MY_MODEL_H
#define MY_MODEL_H

#include "opencv2/opencv.hpp"
#include "outer_model/model_params.hpp"
#include "common.h"

int inference_retinanet_model(rknn_app_context_t *app_ctx, cv::Mat src_img, object_detect_result_list *out_result, const int num_class, const char* model_name);

int inference_retinaface_model(rknn_app_context_t *app_ctx, cv::Mat src_img, retinaface_result *out_result);

int inference_classify_model(rknn_app_context_t *app_ctx, cv::Mat src_img, rknn_output* outputs);

int quick_sort_indice_inverse(float *input, int left, int right, int *indices);
int nms(int validCount, float *outputLocations, int order[], float threshold);

#endif // MY_MODEL_H