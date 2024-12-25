#ifndef _RKNN_DET_CLS_FUNC_H_
#define _RKNN_DET_CLS_FUNC_H_

#include "model_params.hpp"
#include "common.h"

ssd_det_result inference_phone_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

ssd_det_result inference_header_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
int init_retinanet_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_retinanet_model(rknn_app_context_t *app_ctx);
// int inference_retinanet_model(rknn_app_context_t *app_ctx, cv::Mat src_img, ssd_det_result *out_result, const int num_class);

retinaface_result inference_face_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
int init_retinaface_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_retinaface_model(rknn_app_context_t *app_ctx);
// int inference_retinaface_model(rknn_app_context_t *app_ctx, image_buffer_t *img, retinaface_result *out_result);


int init_classify_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_classify_model(rknn_app_context_t *app_ctx);
face_attr_cls_object inference_face_attr_model(rknn_app_context_t *app_ctx, det_model_input input_data, box_rect header_box, bool enable_logger);

object_detect_result_list inference_person_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
int init_yolov10_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolov10_model(rknn_app_context_t* app_ctx);

// resnet
int init_resnet_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_resnet_model(rknn_app_context_t* app_ctx);
// int inference_resnet_model(rknn_app_context_t* app_ctx, image_buffer_t* img, resnet_result* out_result, int topK);
resnet_result inference_rec_person_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

#endif // _RKNN_DET_CLS_FUNC_H_