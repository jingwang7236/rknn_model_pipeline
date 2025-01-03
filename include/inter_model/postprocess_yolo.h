#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "common.h"
#include "image_utils.h"
#include "rknn_api.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.8
#define BOX_THRESH 0.5
#define PROTO_CHANNEL (32)
#define PROTO_HEIGHT (160)
#define PROTO_WEIGHT (160)

enum ModelType {
  UNKNOWN = 0,
  SEGMENT = 1,
  DETECTION = 2,
  OBB = 3,
  POSE = 4,
  V10_DETECTION = 5,
};

typedef struct {
  float kpt[34];
  float visibility[17];
} object_pose_result;

typedef struct {
  int x;
  int y;
  int w;
  int h;
  float theta;
} image_xywht_t;

typedef struct {
  image_xywht_t box;
  float prop;
  int cls_id;
} object_obb_result;

typedef struct {
  int left;
  int top;
  int right;
  int bottom;
} image_rect_t;

typedef struct {
  image_rect_t box;
  float prop;
  int cls_id;
} object_detect_result;

typedef struct {
  uint8_t *seg_mask;
} object_segment_result;

typedef struct {
  int id;
  int count;
  ModelType model_type;
  object_detect_result results[OBJ_NUMB_MAX_SIZE];
  object_segment_result results_seg[OBJ_NUMB_MAX_SIZE];
  object_obb_result results_obb[OBJ_NUMB_MAX_SIZE];
  object_pose_result results_pose[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;
  int model_channel;
  int model_width;
  int model_height;
  bool is_quant;
} rknn_app_context_t;


int init_post_process(std::string &label_path);
void deinit_post_process();

const char *coco_cls_to_name(int cls_id);

int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results);
int post_process_v10_detection(rknn_app_context_t *app_ctx,
                               rknn_output *outputs,
                               letterbox_t *letter_box,
                               float conf_threshold,
                               object_detect_result_list *od_results);
int post_process_obb(rknn_app_context_t *app_ctx, rknn_output *outputs,
                     letterbox_t *letter_box, float conf_threshold,
                     float nms_threshold,
                     object_detect_result_list *od_results);
int post_process_seg(rknn_app_context_t *app_ctx, rknn_output *outputs,
                     letterbox_t *letter_box, float conf_threshold,
                     float nms_threshold,
                     object_detect_result_list *od_results);
int post_process_pose(rknn_app_context_t *app_ctx, rknn_output *outputs,
                      letterbox_t *letter_box, float conf_threshold,
                      float nms_threshold,
                      object_detect_result_list *od_results);
int clamp(float val, int min, int max);
