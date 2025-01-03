#include <stdio.h>
#include <iostream>
#include <string.h>

#include "model_func.hpp"
#include "model_params.hpp"
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
/*-------------------------------------------
                Main  Functions
-------------------------------------------*/
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_name> <image_path>\n", argv[0]);
        return -1;
    }
    int ret;
    const char *model_name = argv[1];
    const char *image_path = argv[2];
    // Load image
    int width, height, channel;
    unsigned char* data = stbi_load(image_path, &width, &height, &channel, 3);
    if (data == NULL) {
        printf("Failed to load image from path: %s\n", image_path);
        return -1;
    }

    // init input data
    det_model_input input_data;
    input_data.data = data;
    input_data.width = width;
    input_data.height = height;
    input_data.channel = channel;

    // header det model
    if (std::string(model_name) == "header_det"){
        const char* model_path = "model/HeaderDet.rknn";
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rknn_app_ctx);  // 初始化
        if (ret != 0) {
            printf("init_retinanet_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }
        ssd_det_result result = inference_header_det_model(&rknn_app_ctx, input_data, true); //推理
        ret = release_model(&rknn_app_ctx);  //释放
        if (ret != 0) {
            printf("release_retinanet_model fail! ret=%d\n", ret);
            return -1;
        }
    }else if(std::string(model_name) == "phone_det"){
        const char* model_path = "model/PhoneDet.rknn";
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rknn_app_ctx);  // 初始化
        if (ret != 0) {
            printf("init_retinanet_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }
        ssd_det_result result = inference_phone_det_model(&rknn_app_ctx, input_data, true); //推理
        ret = release_model(&rknn_app_ctx);  //释放
        if (ret != 0) {
            printf("release_retinanet_model fail! ret=%d\n", ret);
            return -1;
        }
    }else if(std::string(model_name) == "face_det"){
        const char* model_path = "model/RetinaFace.rknn";
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rknn_app_ctx);  // 初始化
        if (ret != 0) {
            printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }
        retinaface_result result = inference_face_det_model(&rknn_app_ctx, input_data, true); //推理
        ret = release_model(&rknn_app_ctx);
        if (ret != 0) {
            printf("release_retinaface_model fail! ret=%d\n", ret);
            return -1;
    }
    }else if (std::string(model_name) == "person_det"){
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "model/yolov10s.rknn";
        ret = init_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov10_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }
        object_detect_result_list result = inference_person_det_model(&rknn_app_ctx, input_data, true); //推理
        ret = release_model(&rknn_app_ctx);
        if (ret != 0)
        {
            printf("release_yolov10_model fail! ret=%d\n", ret);
        }
    }
    else if (std::string(model_name) == "face_attr"){
        // 检测初始化
        const char* det_model_path = "model/HeaderDet.rknn";
        rknn_app_context_t det_rknn_app_ctx;
        memset(&det_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(det_model_path, &det_rknn_app_ctx);  
        // 分类初始化
        rknn_app_context_t cls_rknn_app_ctx;
        memset(&cls_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* cls_model_path = "model/FaceAttr.rknn";
        ret = init_model(cls_model_path, &cls_rknn_app_ctx);
        ssd_det_result det_result = inference_header_det_model(&det_rknn_app_ctx, input_data, true); //头肩检测模型推理
        det_result.count = det_result.count;
        for (int i = 0; i < det_result.count; ++i) {
            box_rect header_box;  // header的box
            header_box.left = std::max(det_result.object[i].box.left, 0);
            header_box.top = std::max(det_result.object[i].box.top, 0);
            header_box.right = std::min(det_result.object[i].box.right, width);
            header_box.bottom = std::min(det_result.object[i].box.bottom, height);
            // 人脸属性模型
            face_attr_cls_object cls_result = inference_face_attr_model(&cls_rknn_app_ctx, input_data, header_box, true);
        }
        ret = release_model(&det_rknn_app_ctx);  //释放
        ret = release_model(&cls_rknn_app_ctx);
    }else if (std::string(model_name) == "ppocr"){
        const char* det_model_path = "model/ppocrv4_det.rknn";
        const char* rec_model_path = "model/ppocrv4_rec.rknn";
        ppocr_system_app_context rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(ppocr_system_app_context));
        ret = init_model(det_model_path, &rknn_app_ctx.det_context);
        ret = init_model(rec_model_path, &rknn_app_ctx.rec_context);
        ppocr_text_recog_array_result_t results = inference_ppocr_det_rec_model(&rknn_app_ctx, input_data, true);
        ret = release_model(&rknn_app_ctx.det_context);
        ret = release_model(&rknn_app_ctx.rec_context);
    }
    else {
        std::cerr << "Unknown model_name: " << model_name << std::endl;
        throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
    }
    stbi_image_free(data);
    return 0;
}
