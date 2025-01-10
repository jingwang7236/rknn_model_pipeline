#include <iostream>
#include <chrono>
#include <string.h>

#include "yolo.h"
#include "yolo_postprocess.h"
#include "file_utils.h"

const int RK3588 = 3;

// 自定义实现兼容 C++11 的 make_unique 模板函数
namespace std {
    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

// 设置模型需要绑定的核心
// Set the core of the model that needs to be bound
int get_core_num() {
    static int core_num = 0;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    int temp = core_num % RK3588;
    core_num++;
    return temp;
}

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int Yolo::init_model(rknn_context *ctx_in, bool copy_weight) {
    int ret = 0;
    int model_len = 0;
    char *model;
    
    // Load RKNN Model
    model_len = read_data_from_file(model_path_.c_str(), &model);

    if (model == nullptr) {
        printf("ERROR: load model fail!\n");
        return -1;
    }
    if (copy_weight) {
        if (get_enable_logger())
            printf("INFO: rknn_dup_context() is called\n");
        // 复用模型参数
        ret = rknn_dup_context(ctx_in, &ctx_);
        if (ret != RKNN_SUCC) {
            printf("ERROR: rknn_dup_context failed! ret = %d\n", ret);
            return -1;
        }
    } else {
        if (get_enable_logger())
            printf("INFO: rknn_init() is called\n");
        
        // 指定的rknn_context 和类内的指向同一块资源
        ret = rknn_init(ctx_in, model, model_len, 0, NULL);
        ctx_ = *ctx_in;

        free(model);
        if (ret != RKNN_SUCC) {
            printf("ERROR: rknn_init fail! ret = %d\n", ret);
            return -1;
        }
    }

    // 绑定模型运行的NPU核心
    int core_num = -1;
    if (core_num < 0 or core_num > RK3588){
        core_num = get_core_num();
    }

    rknn_core_mask core_mask;
    switch (core_num) {
        case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
        case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
        case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }

    if (get_enable_logger()){
        printf("INFO: model run in core %d\n", core_num);
    }

    ret = rknn_set_core_mask(ctx_, core_mask);
    if (ret < 0) {
        printf("ERROR: rknn_set_core_mask failed! error code = %d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("ERROR: get rknn sdk version failed! error code = %d\n", ret);
        return -1;
    }

    if (get_enable_logger())
        printf("INFO: sdk version: %s, driver version: %s\n", version.api_version, version.drv_version);

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("ERROR: rknn_query fail! ret = %d\n", ret);
        return -1;
    }

    // Get Model Input Info
    rknn_tensor_attr input_attrs[io_num.n_input];  //这里使用的是变长数组，不建议这么使用
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            if (get_enable_logger())
                printf("ERROR: input rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));

    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            if (get_enable_logger())
                printf("ERROR: output rknn_query fail! error code = %d\n", ret);
            return -1;
        }
    }

    // 封装初始化
    memset(&ctx_, 0, sizeof(ctx_));

    // Set to context
    app_ctx_.rknn_ctx = ctx_;
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8) {
        app_ctx_.is_quant = true;
    } else {
        app_ctx_.is_quant = false;
    }
    app_ctx_.io_num = io_num;
    app_ctx_.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx_.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx_.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx_.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // printf("model is NCHW input fmt\n");
        app_ctx_.model_channel = input_attrs[0].dims[1];
        app_ctx_.model_height = input_attrs[0].dims[2];
        app_ctx_.model_width = input_attrs[0].dims[3];
    } else {
        // printf("model is NHWC input fmt\n");
        app_ctx_.model_height = input_attrs[0].dims[1];
        app_ctx_.model_width = input_attrs[0].dims[2];
        app_ctx_.model_channel = input_attrs[0].dims[3];
    }
    // printf("model input height=%d, width=%d, channel=%d\n", app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int Yolo::init_model(rknn_app_context_t *app_ctx, bool copy_weight){
    app_ctx_ = *app_ctx;

    int ret;
    int model_len = 0;
    char *model;

    // Load RKNN Model
    model_len = read_data_from_file(model_path_.c_str(), &model);

    if (model == NULL) {
        if (get_enable_logger())
            printf("ERROR: load model fail!\n");
        return -1;
    }
    if (copy_weight) {
        if (get_enable_logger())
            printf("INFO: rknn_dup_context() is called\n");
        // 复用模型参数
        ret = rknn_dup_context(&app_ctx_.rknn_ctx, &ctx_);
        if (ret != RKNN_SUCC) {
            if (get_enable_logger())
                printf("ERROR: rknn_dup_context failed! ret = %d\n", ret);
            return -1;
        }
    } else {
        if (get_enable_logger())
            printf("INFO: rknn_init() is called\n");

        // 指定的rknn_app_context_t 和类内的指向同一块资源
        ret = rknn_init(&ctx_, model, model_len, 0, NULL);
        app_ctx_.rknn_ctx = ctx_;

        free(model);
        if (ret != RKNN_SUCC) {
            printf("ERROR: rknn_init fail! ret = %d\n", ret);
            return -1;
        }
    }

    // 绑定模型运行的NPU核心
    int core_num = -1;
    if (core_num < 0 or core_num > RK3588){
        core_num = get_core_num();
    }

    rknn_core_mask core_mask;
    switch (get_core_num()) {
        case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
        case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
        case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }

    if (get_enable_logger()){
        printf("INFO: model run in core %d\n", core_num);
    }

    ret = rknn_set_core_mask(ctx_, core_mask);
    if (ret < 0) {
        printf("ERROR: rknn_set_core_mask failed! error code = %d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("ERROR: get rknn sdk version failed! error code = %d\n", ret);
        return -1;
    }
    if (get_enable_logger())
        printf("INFO: sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        if (get_enable_logger())
            printf("rknn_query fail! ret = %d\n", ret);
        return -1;
    }

    // Get Model Input Info
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            if (get_enable_logger())
                printf("rknn_query fail! ret = %d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));

    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            if (get_enable_logger())
                printf("output rknn_query fail! error code = %d\n", ret);
            return -1;
        }
    }

    // Set to context (input struct)
    app_ctx_.rknn_ctx = ctx_;
    app_ctx_.io_num = io_num;
    app_ctx_.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx_.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx_.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx_.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // printf("model is NCHW input fmt\n");
        app_ctx_.model_channel = input_attrs[0].dims[1];
        app_ctx_.model_height  = input_attrs[0].dims[2];
        app_ctx_.model_width   = input_attrs[0].dims[3];
    } else {
        // printf("model is NHWC input fmt\n");
        app_ctx_.model_height  = input_attrs[0].dims[1];
        app_ctx_.model_width   = input_attrs[0].dims[2];
        app_ctx_.model_channel = input_attrs[0].dims[3];
    }

    return 0;
}

Yolo::~Yolo() { release_model(); }

int Yolo::release_model() {
  if (app_ctx_.input_attrs != nullptr) {
    if (get_enable_logger())
        printf("INFO: free input_attrs by Class\n");
    free(app_ctx_.input_attrs);
  }
  if (app_ctx_.output_attrs != nullptr) {
    if (get_enable_logger())
        printf("INFO: free output_attrs by Class\n");
    free(app_ctx_.output_attrs);
  }
  if (app_ctx_.rknn_ctx != 0) {
    if (get_enable_logger())
        printf("INFO: rknn_destroy by Class\n");
    rknn_destroy(app_ctx_.rknn_ctx);
    app_ctx_.rknn_ctx = 0;
  }
  return 0;
}

int Yolo::release_model(rknn_app_context_t *app_ctx) {
    if (app_ctx->input_attrs != NULL) {
        printf("INFO: free input_attrs by Struct\n");
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        printf("INFO: free output_attrs by Struct\n");
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        printf("INFO: rknn_destroy by Struct\n");
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

void Yolo::set_hyperparamter(int obj_class_num, int kpt_num, int result_num){
    obj_class_num_ = obj_class_num;
    kpt_num_ = kpt_num;
    result_num_ = result_num;
}

rknn_context *Yolo::get_rknn_context() { return &(this->ctx_); }

rknn_app_context_t *Yolo::get_rknn_app_context_t() { return &(this->app_ctx_); }

int Yolo::inference_yolo_model(void *image_buf, void *od_results, letterbox_t letter_box) {
    // 模型开始推理时间戳
    auto total_start_time = std::chrono::high_resolution_clock::now();

    rknn_input inputs[app_ctx_.io_num.n_input];
    rknn_output outputs[app_ctx_.io_num.n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].buf = image_buf;

    int ret = rknn_inputs_set(app_ctx_.rknn_ctx, app_ctx_.io_num.n_input, inputs);
    if (ret < 0) {
        printf("ERROR: rknn_input_set failed! error code = %d\n", ret);
        return -1;
    }

    ret = rknn_run(app_ctx_.rknn_ctx, nullptr);
    if (ret != RKNN_SUCC) {
        printf("ERROR: rknn_run failed, error code = %d\n", ret);
        return -1;
    }

    for (int i = 0; i < app_ctx_.io_num.n_output; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx_.is_quant);
    }

    outputs_lock_.lock();
    ret = rknn_outputs_get(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, outputs, nullptr);

    if (ret != RKNN_SUCC) {
        printf("ERROR: rknn_outputs_get failed, error code = %d\n", ret);
        return -1;
    }

    const float nms_threshold = nms_thresh_;         // 默认的NMS阈值
    const float box_conf_threshold = box_thresh_;    // 默认的置信度阈值

    // 推理结束时间
    auto inference_end_time = std::chrono::high_resolution_clock::now();

    // 后处理，根据 model_type_ 调用不同的处理函数，将 od_results 从 void* 转换为目标数据类型，清零结构体并设置类型
    if (model_type_ == YoloModelType::DETECTION) {
        auto* detect_results = static_cast<object_detect_result_list*>(od_results);
        memset(detect_results, 0, sizeof(object_detect_result_list));
        post_process_hw(&app_ctx_, outputs, &letter_box, box_conf_threshold, nms_threshold, detect_results, 
                        obj_class_num_);
    } 
    else if (model_type_ == YoloModelType::POSE) {
        auto* pose_results = static_cast<object_detect_pose_result_list*>(od_results);
        memset(pose_results, 0, sizeof(object_detect_pose_result_list));
        post_process_pose_hw(&app_ctx_, outputs, &letter_box, box_conf_threshold, nms_threshold, pose_results, 
                            obj_class_num_, kpt_num_, result_num_);
    } 
    else if (model_type_ == YoloModelType::OBB) {
        auto* obb_results = static_cast<object_detect_obb_result_list*>(od_results);
        memset(obb_results, 0, sizeof(object_detect_obb_result_list));
        post_process_obb_hw(&app_ctx_, outputs, &letter_box, box_conf_threshold, nms_threshold, obb_results);
    }
    // else if (model_type_ == YoloModelType::V10_DETECTION){
    // }
    else {
        if (get_enable_logger())
            printf("Unsupported model type!\n");
        return -1;
    }

    // 释放 rknn 输出
    rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, outputs);
    outputs_lock_.unlock();

    // 计算时间
    if (get_enable_logger()){
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration<double, std::milli>(inference_end_time - total_start_time);
        auto postprocess_duration = std::chrono::duration<double, std::milli>(total_end_time - inference_end_time);
        auto total_duration = std::chrono::duration<double, std::milli>(total_end_time - total_start_time);

        printf("Total infer time %.1f ms: Model time is %.1f ms and PostProcess time is %.1fms\n",
            total_duration.count(), inference_duration.count(), postprocess_duration.count());
    }
    
    return 0;
}
