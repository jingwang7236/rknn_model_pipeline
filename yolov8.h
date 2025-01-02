
#ifndef _RKNN_DEMO_YOLOV8_H_
#define _RKNN_DEMO_YOLOV8_H_

// #include "rknn_api.h"
// #include "common.h"
#include "image_utils.h"
#include "opencv2/opencv.hpp"
#include "outer_model/model_params.hpp"



int init_post_process();
int init_post_process(const char* label_txt_path);
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
char* coco_cls_to_name(int cls_id, int obj_class_num);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
int post_process_obb(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_obb_result_list* od_results);
int post_process_pose(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_pose_result_list* od_results);

void deinitPostProcess();

int inference_yolov8_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results);
int inference_yolov8_obb_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_obb_result_list* od_results);
int inference_yolov8_pose_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_pose_result_list* od_results);

#endif //_RKNN_DEMO_YOLOV8_H_