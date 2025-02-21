#ifndef _RKNN_RESNET_H_
#define _RKNN_RESNET_H_

#include "outer_model/model_params.hpp"
#include "common.h"
#include "opencv2/opencv.hpp"

// int init_resnet_model(const char* model_path, rknn_app_context_t* app_ctx);
// int release_resnet_model(rknn_app_context_t* app_ctx);
int inference_resnet_model(rknn_app_context_t* app_ctx, image_buffer_t* img, resnet_result* out_result, int topK);
int inference_resnet_model_opencv(rknn_app_context_t* app_ctx, void* image_buf, resnet_result* out_result, int topk);

#endif //_RKNN_RESNET_H_