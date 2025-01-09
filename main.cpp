#include <stdio.h>
#include <iostream>
#include <string.h>

#include "model_func.hpp"
#include "model_params.hpp"
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void print_rknn_app_context(const rknn_app_context_t& ctx) {
    std::cout << "rknn_ctx: " << ctx.rknn_ctx << std::endl;
    std::cout << "model_channel: " << ctx.model_channel << std::endl;
    std::cout << "model_width: " << ctx.model_width << std::endl;
    std::cout << "model_height: " << ctx.model_height << std::endl;
    std::cout << "is_quant: " << (ctx.is_quant ? "true" : "false") << std::endl;
}

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
    }
    else if(std::string(model_name) == "phone_det"){
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
    }
    else if(std::string(model_name) == "face_det"){
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
    }
    else if (std::string(model_name) == "person_det"){
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
    }
    else if (std::string(model_name) == "det_knife"){
        model_inference_params params_det_knife = { 640,640,0.6f,0.25f };
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "model/yolov8n_1105_det_knife_i8.rknn";
        //const char* label_txt_path = "model/classes_knife.txt";
        ret = init_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        rknn_app_ctx.is_quant = true;

        // print_rknn_app_context(rknn_app_ctx);
        
        object_detect_result_list result = inference_det_knife_model(&rknn_app_ctx, input_data, params_det_knife, false, true); //推理
        ret = release_model(&rknn_app_ctx);
        if (ret != 0)
        {
            printf("release_yolov8_model fail! ret=%d\n", ret);
        }
    }
    else if (std::string(model_name) == "det_gun") {

        /* 推理参数 width height nms_ths box_ths*/
        model_inference_params params_det_gun = { 640,640,0.6f,0.25f };
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "model/jhpoc_yv8s_1212_det_gun_640_i8.rknn";
        // const char* label_txt_path = "model/classes_gun.txt";
        ret = init_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        rknn_app_ctx.is_quant = true;

        //print_rknn_app_context(rknn_app_ctx);

        object_detect_result_list result = inference_det_gun_model(&rknn_app_ctx, input_data, params_det_gun, false, true); //推理
        ret = release_model(&rknn_app_ctx);
        if (ret != 0)
        {
            printf("release_yolov8_model fail! ret=%d\n", ret);
        }
    }
    else if (std::string(model_name) == "det_stat_door") {
        model_inference_params params_det_stat_door = { 640,640,0.6f,0.25f };
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "../model/jhpoc_1225_stat_door_det2_640_rk.rknn";
        //const char* label_txt_path = "model/classes_door.txt";
        ret = init_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        rknn_app_ctx.is_quant = false;

        print_rknn_app_context(rknn_app_ctx);

        object_detect_result_list result = inference_det_stat_door_model(&rknn_app_ctx, input_data, params_det_stat_door, false, true); //推理
        ret = release_model(&rknn_app_ctx);
        if (ret != 0)
        {
            printf("release_yolov8_model fail! ret=%d\n", ret);
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
    }
    else if (std::string(model_name) == "ppocr"){
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
    else if(std::string(model_name) == "rec_ren"){
        // 分类初始化
        const char* model_path = "model/rec_ren_1225_resnet18.rknn";
        rknn_app_context_t rec_rknn_app_ctx;
        memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rec_rknn_app_ctx);  
        
        if (ret != 0)
        {
            printf("init_rec_ren_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        resnet_result rec_result = inference_rec_person_resnet18_model(&rec_rknn_app_ctx, input_data, false);
        // std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
        ret = release_model(&rec_rknn_app_ctx);
    }
    else if(std::string(model_name) == "det_hand"){
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "model/det_hand_s_25_01_02.rknn";
        ret = init_model(model_path, &rknn_app_ctx);

        if (ret != 0){
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        rknn_app_ctx.is_quant = true;
        // print_rknn_app_context(rknn_app_ctx);
        
        object_detect_result_list result = inference_det_hand_model(&rknn_app_ctx, input_data, false, false); //推理

        ret = release_model(&rknn_app_ctx);
        
        if (ret != 0){
            printf("release_yolov8_model fail! ret=%d\n", ret);
        }
    }
    else if(std::string(model_name) == "det_kx"){
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "model/det_kx_s_24_12_18.rknn";
        ret = init_model(model_path, &rknn_app_ctx);

        if (ret != 0){
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        rknn_app_ctx.is_quant = true;
        // print_rknn_app_context(rknn_app_ctx);
        
        object_detect_result_list result = inference_det_kx_model(&rknn_app_ctx, input_data, false); //推理

        ret = release_model(&rknn_app_ctx);
        
        if (ret != 0){
            printf("release_yolov8_model fail! ret=%d\n", ret);
        }
    }
    else if(std::string(model_name) == "rec_hand"){
        // 分类初始化
        const char* model_path = "model/rec_hand_1216_resnet18.rknn";
        rknn_app_context_t rec_rknn_app_ctx;
        memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rec_rknn_app_ctx);  
        
        if (ret != 0)
        {
            printf("init rec_hand model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        resnet_result rec_result = inference_rec_hand_resnet18_model(&rec_rknn_app_ctx, input_data, false);
        // std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
        ret = release_model(&rec_rknn_app_ctx);
    }
    else if(std::string(model_name) == "pose_ren"){
        // 分类初始化
        const char* model_path = "model/yolov8s-pose.rknn";
        rknn_app_context_t rec_rknn_app_ctx;
        memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rec_rknn_app_ctx);  
        
        if (ret != 0){
            printf("init pose_ren model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        object_detect_pose_result_list pose_result = inference_pose_ren_model(&rec_rknn_app_ctx, input_data, false);
        ret = release_model(&rec_rknn_app_ctx);
    }
    else if(std::string(model_name) == "rec_kx_orient"){
        // 分类初始化
        const char* model_path = "model/rec_kx_orient_1207_resnet18.rknn";
        rknn_app_context_t rec_rknn_app_ctx;
        memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rec_rknn_app_ctx);  
        
        if (ret != 0)
        {
            printf("init rec_kx_orient model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        resnet_result rec_result = inference_rec_kx_orient_resnet18_model(&rec_rknn_app_ctx, input_data, false);
        // std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
        ret = release_model(&rec_rknn_app_ctx);
    }
    //else if(std::string(model_name) == "pose_kx_hp"){
    //    // 分类初始化
    //    const char* model_path = "model/pose_kx_hp_s_24_12_12.rknn";
    //    rknn_app_context_t rec_rknn_app_ctx;
    //    memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    //    ret = init_model(model_path, &rec_rknn_app_ctx);  
    //    
    //    if (ret != 0){
    //        printf("init pose_kx_hp model fail! ret=%d model_path=%s\n", ret, model_path);
    //        return -1;
    //    }

    //    object_detect_pose_result_list pose_result = inference_pose_kx_hp_model(&rec_rknn_app_ctx, input_data, false);
    //    ret = release_model(&rec_rknn_app_ctx);
    //}
    //else if(std::string(model_name) == "pose_kx_sz"){
    //    // 分类初始化
    //    const char* model_path = "model/pose_kx_sz_s_24_12_07.rknn";
    //    rknn_app_context_t rec_rknn_app_ctx;
    //    memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    //    ret = init_model(model_path, &rec_rknn_app_ctx);  
    //    
    //    if (ret != 0){
    //        printf("init pose_kx_sz model fail! ret=%d model_path=%s\n", ret, model_path);
    //        return -1;
    //    }

    //    object_detect_pose_result_list pose_result = inference_pose_kx_sz_model(&rec_rknn_app_ctx, input_data, false);
    //    ret = release_model(&rec_rknn_app_ctx);
    //}
    else if (std::string(model_name) == "obb_stick") {

        /* 推理参数 width height nms_ths box_ths*/
        model_inference_params params_det_gun = { 1024,1024,0.6f,0.25f };
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "../model/jhpoc_1206-test1_obb_stick_1024_i8.rknn";
        ret = init_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        object_detect_pose_result_list pose_result = inference_pose_kx_sz_model(&rec_rknn_app_ctx, input_data, false);
        ret = release_model(&rec_rknn_app_ctx);
    }
    else if (std::string(model_name) == "obb_stick") {

        /* 推理参数 width height nms_ths box_ths*/
        model_inference_params params_det_gun = { 1024,1024,0.6f,0.25f };
        rknn_app_context_t rknn_app_ctx;
        memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        const char* model_path = "../model/jhpoc_250109-test1_obb_stick_1024.rknn";
        ret = init_model(model_path, &rknn_app_ctx);
        if (ret != 0)
        {
            printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        rknn_app_ctx.is_quant = false;

        //print_rknn_app_context(rknn_app_ctx);

        object_detect_obb_result_list result = inference_obb_stick_model(&rknn_app_ctx, input_data, params_det_gun, false, true); //推理
        ret = release_model(&rknn_app_ctx);
        if (ret != 0)
        {
            printf("release_yolov8_model fail! ret=%d\n", ret);
        }
    }

    else if (std::string(model_name) == "rec_stat_door") {
        // 分类初始化
        const char* model_path = "../model/cls_stat_door_model_resnet18_150108_i8.rknn";
        rknn_app_context_t rec_rknn_app_ctx;
        memset(&rec_rknn_app_ctx, 0, sizeof(rknn_app_context_t));
        ret = init_model(model_path, &rec_rknn_app_ctx);

        if (ret != 0)
        {
            printf("init_rec_ren_model fail! ret=%d model_path=%s\n", ret, model_path);
            return -1;
        }

        resnet_result rec_result = inference_rec_stat_door_resnet18_model(&rec_rknn_app_ctx, input_data, false);
        std::cout << "Class index: " << rec_result.cls << ", Score: " << rec_result.score << std::endl;
        ret = release_model(&rec_rknn_app_ctx);
        }
    else {
        std::cerr << "Unknown model_name: " << model_name << std::endl;
        throw std::invalid_argument("Unknown model_name: " + std::string(model_name));
    }
    stbi_image_free(data);
    return 0;
}
