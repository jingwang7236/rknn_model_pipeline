// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <iostream>

#include "image_utils.h"
#include "image_drawing.h"
#include "outer_model/model_func.hpp"
#include "inter_model/retinanet.hpp"

/*-------------------------------------------
                  Functions
-------------------------------------------*/
retinaface_result inference_face_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger=false)
{
    retinaface_result result;

    // cv::Mat orig_img_rgb;
    // cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);
    // cv::Mat orig_img 图片转为image_buffer_t格式
    /*
    unsigned char* data = input_data.data;
    int width = input_data.width;
    int height = input_data.height;
    int channel = input_data.channel;
    */
    image_buffer_t src_img;
    memset(&src_img, 0, sizeof(image_buffer_t));
    src_img.width = input_data.width;
    src_img.height = input_data.height;
    src_img.format = IMAGE_FORMAT_RGB888;
    src_img.size = input_data.width * input_data.height * input_data.channel;
    src_img.virt_addr = (unsigned char *)malloc(src_img.size);

    if (src_img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", src_img.size);
        return result;
    }
    memcpy(src_img.virt_addr, input_data.data, src_img.size);

    rknn_context ctx = 0;
    int            ret;
    int            model_len = 0;
    unsigned char* model;

    float face_det_threshold = 0.2;
    // const char* model_path = "model/RetinaFace.rknn";

    // rknn_app_context_t rknn_app_ctx;
    // memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // ret = init_retinaface_model(model_path, &rknn_app_ctx);
    // if (ret != 0) {
    //     printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, model_path);
    //     return result;
    // }

    ret = inference_retinaface_model(app_ctx, &src_img, &result);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d\n", ret);
        return result;
    }
    if (enable_logger) {
        printf("detect result num: %d\n", result.count);
        for (int i = 0; i < result.count; ++i) {
            if (result.object[i].score < face_det_threshold) {
                continue;
            }
            printf("face @(%d %d %d %d) score=%f\n", result.object[i].box.left, result.object[i].box.top,
                result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
        }
    }
    bool enable_draw_image = false; //画图
    if (enable_draw_image) {
        for (int i = 0; i < result.count; ++i) {
            if (result.object[i].score < face_det_threshold) {
                continue;
            }
            int rx = result.object[i].box.left;
            int ry = result.object[i].box.top;
            int rw = result.object[i].box.right - result.object[i].box.left;
            int rh = result.object[i].box.bottom - result.object[i].box.top;
            draw_rectangle(&src_img, rx, ry, rw, rh, COLOR_GREEN, 3);
            char score_text[20];
            snprintf(score_text, 20, "%0.2f", result.object[i].score);
            
            draw_text(&src_img, score_text, rx, ry, COLOR_RED, 20);
            for(int j = 0; j < 5; j++) {
                draw_circle(&src_img, result.object[i].ponit[j].x, result.object[i].ponit[j].y, 2, COLOR_ORANGE, 4);
            }
        }
        const char* image_path = "face_det_result.png";
        write_image(image_path, &src_img);
        std::cout << "Draw result on" << image_path << " is finished." << std::endl;
    }
    
    
    // ret = release_retinaface_model(&rknn_app_ctx);
    // if (ret != 0) {
    //     printf("release_retinaface_model fail! ret=%d\n", ret);
    // }
    if (src_img.virt_addr != NULL) {
        free(src_img.virt_addr);
    }
    return result;
}

