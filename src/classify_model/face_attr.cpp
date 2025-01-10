// backbone:repvgg, 3.2M params, person attribute,include: hat/hellem, glass, mask


/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <iostream>
#include "opencv2/opencv.hpp"

#include "model_params.hpp"
#include "model_func.hpp"
#include "retinanet.hpp"
#include <chrono>  // 添加这一行
using namespace std::chrono;  // 添加这一行
/*-------------------------------------------
                  Functions
-------------------------------------------*/

cls_model_result inference_face_attr_model(rknn_app_context_t *app_ctx, det_model_input input_data, box_rect header_box, bool enable_logger=false)
{
    cls_model_result result;
    unsigned char* data = input_data.data; // scene image
    int width = input_data.width;
    int height = input_data.height;
    int channel = input_data.channel;
    if (enable_logger){
        printf("Image size: %d x %d x %d\n", width, height, channel);
    }
    cv::Mat cv_img(height, width, CV_8UC3, data);
    // cv::Mat orig_img(height, width, CV_MAKETYPE(CV_8U, channels), input_data);
    if (cv_img.empty()) {
        std::cerr << "Image is empty or invalid." << std::endl;
    }
    // orig_img的通道顺序为cv图片的默认顺序bgr
    cv::Mat orig_img;
    cv::cvtColor(cv_img, orig_img, cv::COLOR_RGB2BGR); 
    // header crop img
    // auto start = std::chrono::high_resolution_clock::now();
    cv::Mat header_crop_img = orig_img(cv::Rect(header_box.left, header_box.top, header_box.right - header_box.left, header_box.bottom - header_box.top));
    // auto end = std::chrono::high_resolution_clock::now();
    // printf("crop img cost time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
    rknn_context ctx = 0;
    int            ret;
    int            model_len = 0;
    unsigned char* model;
   // 动态分配outputs数组
    rknn_output* outputs = (rknn_output*)malloc(app_ctx->io_num.n_output * sizeof(rknn_output));
    if (outputs == NULL) {
        printf("malloc outputs fail!\n");
        return result;
    }
    ret = inference_classify_model(app_ctx, header_crop_img, outputs);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d\n", ret);
        return result;
    }
    // post_process_classify
    result.num_class = app_ctx->io_num.n_output;
    int attr_num[result.num_class];
    attr_num[0] = FaceAttrModelClass::FACE_ATTR_CLASS_1;
    attr_num[1] = FaceAttrModelClass::FACE_ATTR_CLASS_2;
    attr_num[2] = FaceAttrModelClass::FACE_ATTR_CLASS_3;

    for (int i = 0; i < result.num_class; i++) {
        float *output_data = (float *)outputs[i].buf;
        int max_index = -1;
        float max_value = -0.1;
        for (int j = 0; j < attr_num[i]; j++) {
            if (output_data[j] > max_value) {
                max_value = output_data[j];
                max_index = j;
            }
        }
        result.cls_output[i] = max_index;
    }
    if (enable_logger) {
        printf("face attr result: [%d,%d,%d]\n", result.cls_output[0], result.cls_output[1], result.cls_output[2]);
    }
    rknn_outputs_release(app_ctx->rknn_ctx, 3, outputs);
    free(outputs);
    return result;
}

