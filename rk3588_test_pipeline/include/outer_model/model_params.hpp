#ifndef _RKNN_DET_CLS_H_
#define _RKNN_DET_CLS_H_

// #define MODEL_OK 0      // 模型推理成功
// #define MODEL_ERR 1     // 模型推理失败
// #define INPUT_ERR 2  // 模型输入参数错误
// #define GPU_ERR 3       // GPU资源申请失败
// #define MEM_ERR 4       // 内存申请失败

#define OBJ_MAX_NUM 128

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

//算法的输出结果
typedef struct det_model_result{
    int error_code;
    ssd_det_result det_result;
} det_model_result;

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
#define FACE_ATTR_NUM_CLASS 3
#define FACE_ATTR_CLASS_1 3  // 0: negtive, 1: hat, 2: helmet
#define FACE_ATTR_CLASS_2 2  // 0: negtive, 1: glassess
#define FACE_ATTR_CLASS_3 2  // 0: negtive, 1: mask
typedef struct face_attr_cls_object {
    int num_class;
    int cls_output[FACE_ATTR_NUM_CLASS];
} face_attr_cls_object;  // 人脸属性输出结果



// yolov10 det model

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25


typedef struct {
    box_rect box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

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

#endif //_RKNN_DET_CLS_H_