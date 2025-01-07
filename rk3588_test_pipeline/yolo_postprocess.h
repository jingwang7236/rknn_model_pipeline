#pragma once

#include "image_utils.h"
#include "opencv2/opencv.hpp"
#include "model_params.hpp"

int post_process_det_hw(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results, int obj_class_num=1);
int post_process_obb_hw(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_obb_result_list* od_results, int obj_class_num=1);
int post_process_pose_hw(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_pose_result_list* od_results, int obj_class_num=1, int kpt_num=17);
