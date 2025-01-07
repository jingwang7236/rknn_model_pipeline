#ifndef _RKNN_DET_CLS_FUNC_H_
#define _RKNN_DET_CLS_FUNC_H_

#include "model_params.hpp"
#include "common.h"
#include "ppocr_system.h"

int init_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_model(rknn_app_context_t *app_ctx);

ssd_det_result inference_phone_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

ssd_det_result inference_header_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

retinaface_result inference_face_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

face_attr_cls_object inference_face_attr_model(rknn_app_context_t *app_ctx, det_model_input input_data, box_rect header_box, bool enable_logger);

object_detect_result_list inference_person_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

ppocr_text_recog_array_result_t inference_ppocr_det_rec_model(ppocr_system_app_context *rknn_app_ctx, det_model_input input_data, bool enable_logger);

object_detect_result_list inference_det_knife_model(rknn_app_context_t* app_ctx, det_model_input input_data, char* label_txt_path,bool enable_logger);

object_detect_result_list inference_det_gun_model(rknn_app_context_t* app_ctx, det_model_input input_data, char* label_txt_path,bool enable_logger);

object_detect_result_list inference_det_stat_door_model(rknn_app_context_t* app_ctx, det_model_input input_data, char* label_txt_path, bool enable_logger);

// resnet rec ren
resnet_result inference_rec_person_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger);

// det hand
object_detect_result_list inference_det_hand_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool det_by_square = true, bool enable_logger = true);


#endif // _RKNN_DET_CLS_FUNC_H_
