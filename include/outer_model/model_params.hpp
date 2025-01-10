#ifndef _RKNN_DET_CLS_H_
#define _RKNN_DET_CLS_H_

#include <string>

#include "common.h"
#include <map>
#include <memory>


// #define MODEL_OK 0      // 模型推理成功
// #define MODEL_ERR 1     // 模型推理失败
// #define INPUT_ERR 2  // 模型输入参数错误
// #define GPU_ERR 3       // GPU资源申请失败
// #define MEM_ERR 4       // 内存申请失败

#define OBJ_MAX_NUM 128
#define CLASS_MAX_NUM 80  //分类模型最多类别

// det model
typedef struct box_rect {
    int left;    ///< Most left coordinate
    int top;     ///< Most top coordinate
    int right;   ///< Most right coordinate
    int bottom;  ///< Most bottom coordinate
} box_rect;


// 记录检测模型的结果，包括Box和对应cls和分数
typedef struct det_object_t {
    int cls;     
    box_rect box;  
    float score;
} det_object_t;

typedef struct ssd_det_result{
    int count;
    det_object_t object[OBJ_MAX_NUM];
} ssd_det_result;


//算法输入数据格式
typedef struct det_model_input{
    int width;
    int height;
    int channel;
    unsigned char *data;
} det_model_input;


typedef struct ponit_t {
    int x;
    int y;
} ponit_t;  // add

typedef struct retinaface_object_t {
    int cls;       
    box_rect box;  
    float score;      
    ponit_t ponit[5];
} retinaface_object_t;

typedef struct retinaface_result{
    int count;
    retinaface_object_t object[OBJ_MAX_NUM];
} retinaface_result;  // 比较ssd_det_result多了一个point


// classify model

enum FaceAttrModelClass {
    FACE_ATTR_CLASS_1 = 3,
    FACE_ATTR_CLASS_2 = 2,
    FACE_ATTR_CLASS_3 = 2,
};// 人脸属性多分类模型定义

typedef struct cls_model_result {
    int num_class;
    int cls_output[CLASS_MAX_NUM];
} cls_model_result;  // 分类模型输出结果

// yolo det model

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)


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

typedef struct {
    image_rect_t box;
    float keypoints[17][3];//keypoints x,y,conf
    float prop;
    int cls_id;
} object_detect_pose_result;

typedef struct {
    int id;
    int count;
    object_detect_pose_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_pose_result_list;


typedef struct {
    image_obb_box_t box;
    float prop;
    int cls_id;
} object_detect_obb_result;

typedef struct {
    int id;
    int count;
    object_detect_obb_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_obb_result_list;


//算法输入数据格式
typedef struct rec_model_input{
    int width;
    int height;
    int channel;
    unsigned char *data;
} rec_model_input;

// ResNet
typedef struct {
    int cls;
    float score;
} resnet_result;

typedef struct {
    int cls;
    float score;
} mobilenet_result;


// 计算模型指标需要的结构体
// 定义 InferenceFunction 类型,分类模型返回值是cls_model_result
typedef cls_model_result (*ClsInferenceFunction)(rknn_app_context_t*, det_model_input, box_rect, bool);
// 定义 ModelInfo 结构体,包含模型名、模型路径、推理函数
struct ClsModelInfo {
    std::string modelName;
    std::string modelPath;
    ClsInferenceFunction inferenceFunc;
};

// 定义 InferenceFunction 类型,检测模型返回值是object_detect_result_list
typedef object_detect_result_list (*DetInferenceFunction)(rknn_app_context_t*, det_model_input, bool);
// 定义 ModelInfo 结构体,包含模型名、模型路径、推理函数
struct DetModelInfo {
    std::string modelName;
    std::string modelPath;
    DetInferenceFunction inferenceFunc;
};


/* inference params */
typedef struct model_inference_params {
    int input_width;
    int input_height;
    float nms_threshold;
    float box_threshold;
}model_inference_params;

typedef struct cls_model_inference_params {
    int top_k;
    int img_height;
    int img_width;
}cls_model_inference_params;

/* model classes */

extern std::map<int, std::string> det_gun_category_map; /* {0, "gun"} */
extern std::map<int, std::string> det_knife_category_map; /* {0, "knife"} */
extern std::map<int, std::string> det_stat_door_category_map; /* {0, "closed"},{1, "open"} */
extern std::map<int, std::string> obb_stick_category_map; /* {0, "stick"} */
extern std::map<int, std::string> cls_stat_door_category_map; /* {0, "closed"},{1, "open"},{2, "other"}  not door object */ 

/*-------------------------------------------
            YOLO common start
-------------------------------------------*/
enum YoloModelType {
    UNKNOWN = 0,
    DETECTION = 1,
    OBB = 2,
    POSE = 3,
    V10_DETECTION = 4,
};

/*-------------------------------------------
            YOLO common end
-------------------------------------------*/

#endif //_RKNN_DET_CLS_H_