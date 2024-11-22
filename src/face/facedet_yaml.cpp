/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yaml.h"  // 配置文件解析库

#include "retinaface.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
// 输入图片,输出人脸检测+人脸属性结果
int main(int argc, char **argv) {
    if (argc != 2) {
        printf("%s <image_path>\n", argv[0]);
        return -1;
    }
    const char *image_path = argv[1];
    
    // 加载配置文件
    std::string file_name = "config.yaml";
    YAML::Node config = YAML::LoadFile(file_name);
    if (!config) {
        std::cout << "Open config File:" << file_name << " failed.";
        return false;
    }
    
    std::string face_det_model_path;
    float face_det_threshold;
    // 从配置文件中获取模型路径和阈值
    if (config["FaceDetector"]) {
        if (config["FaceDetector"]["modelpath"])
            face_det_model_path = config["FaceDetector"]["modelpath"].as<std::string>();
        if (config["FaceDetector"]["threshold"])
            face_det_threshold = config["FaceDetector"]["threshold"].as<float>();
        // 输出结果验证
        std::cout << "Model Path: " << face_det_model_path.c_str() << std::endl;
        std::cout << "Threshold: " << face_det_threshold << std::endl;
        
    } else {
        fprintf(stderr, "the node blog doesn't exist\n");
        return -1;
    }
    
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_retinaface_model(face_det_model_path.c_str(), &rknn_app_ctx);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, face_det_model_path.c_str());
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0) {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        return -1;
    }
    
    bool modelSwitch = true;
    retinaface_result result;
    ret = inference_retinaface_model(&rknn_app_ctx, &src_image, &result);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d\n", ret);
        modelSwitch = false;
    }
    
    if (modelSwitch) {
        // 从配置文件中获取开关状态
        bool enable_draw_image = false;
        if (config["enable_draw_image"])
            enable_draw_image = config["enable_draw_image"].as<bool>();
        
        for (int i = 0; i < result.count; ++i) {
            if (enable_draw_image) {
                int rx = result.object[i].box.left;
                int ry = result.object[i].box.top;
                int rw = result.object[i].box.right - result.object[i].box.left;
                int rh = result.object[i].box.bottom - result.object[i].box.top;
                draw_rectangle(&src_image, rx, ry, rw, rh, COLOR_GREEN, 3);
                char score_text[20];
                snprintf(score_text, 20, "%0.2f", result.object[i].score);
                
                draw_text(&src_image, score_text, rx, ry, COLOR_RED, 20);
                for(int j = 0; j < 5; j++) {
                    draw_circle(&src_image, result.object[i].ponit[j].x, result.object[i].ponit[j].y, 2, COLOR_ORANGE, 4);
                }
            }
            // 过滤分数低于face_det_threshold的结果
            if (result.object[i].score < face_det_threshold) {
                continue;
            }
            printf("face @(%d %d %d %d) score=%f\n", result.object[i].box.left, result.object[i].box.top,
                result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
        }
        if (enable_draw_image) {
            write_image("result.jpg", &src_image);
            std::cout << "Draw result on result.jpg is finished." << std::endl;
        }
    }
    ret = release_retinaface_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_retinaface_model fail! ret=%d\n", ret);
    }
    if (src_image.virt_addr != NULL) {
        free(src_image.virt_addr);
    }
    return 0;
}

// TODO: 后续接人脸属性算法的代码