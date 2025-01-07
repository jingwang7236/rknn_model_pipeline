
#ifndef _RKNN_DEMO_YOLOV10_H_
#define _RKNN_DEMO_YOLOV10_H_

// #include "rknn_api.h"
// #include "common.h"
#include "image_utils.h"
#include "opencv2/opencv.hpp"
#include "outer_model/model_params.hpp"

/* 拷贝到outer_model/model_params.hpp文件, 外部调用
#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;
*/


int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

void deinitPostProcess();

// int init_yolov10_model(const char* model_path, rknn_app_context_t* app_ctx);

// int release_yolov10_model(rknn_app_context_t* app_ctx);

int inference_yolov10_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results);
int inference_yolov10_model_opencv(rknn_app_context_t *app_ctx, cv::Mat src_img, object_detect_result_list *od_results);

#endif //_RKNN_DEMO_YOLOV10_H_