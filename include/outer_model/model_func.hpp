#ifndef _RKNN_DET_CLS_FUNC_H_
#define _RKNN_DET_CLS_FUNC_H_

#include "model_params.hpp"
#include "common.h"

int init_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_model(rknn_app_context_t *app_ctx);

ssd_det_result inference_phone_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

ssd_det_result inference_header_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
// int init_retinanet_model(const char *model_path, rknn_app_context_t *app_ctx);
// int release_retinanet_model(rknn_app_context_t *app_ctx);

retinaface_result inference_face_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
// int init_retinaface_model(const char *model_path, rknn_app_context_t *app_ctx);
// int release_retinaface_model(rknn_app_context_t *app_ctx);


// int init_classify_model(const char *model_path, rknn_app_context_t *app_ctx);
// int release_classify_model(rknn_app_context_t *app_ctx);
face_attr_cls_object inference_face_attr_model(rknn_app_context_t *app_ctx, det_model_input input_data, box_rect header_box, bool enable_logger);

object_detect_result_list inference_person_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);
// int init_yolov10_model(const char* model_path, rknn_app_context_t* app_ctx);
// int release_yolov10_model(rknn_app_context_t* app_ctx);

#endif // _RKNN_DET_CLS_FUNC_H_