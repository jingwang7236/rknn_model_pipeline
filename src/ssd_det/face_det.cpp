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
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif
retinaface_result inference_face_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger=false)
{
    int            ret;
    retinaface_result result;
    float face_det_threshold = 0.2;
    unsigned char* data = input_data.data;
    int width = input_data.width;
    int height = input_data.height;
    int channel = input_data.channel;

    cv::Mat cv_img(height, width, CV_8UC3, data);
    if (cv_img.empty()) {
        std::cerr << "Image is empty or invalid." << std::endl;
    }
    // orig_img的通道顺序为cv图片的默认顺序bgr
    cv::Mat orig_img;
    cv::cvtColor(cv_img, orig_img, cv::COLOR_RGB2BGR); 

    ret = inference_retinaface_model(app_ctx, orig_img, &result);
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
        cv::Mat orig_img_clone = orig_img.clone();
        for (int i = 0; i < result.count; ++i) {
            if (result.object[i].score < face_det_threshold) {
                continue;
            }
            int rx = result.object[i].box.left;
            int ry = result.object[i].box.top;
            int rw = result.object[i].box.right - result.object[i].box.left;
            int rh = result.object[i].box.bottom - result.object[i].box.top;
            cv::Rect box(rx, ry, rw, rh);
            std::string text = "face";
            std::string score_str = std::to_string(result.object[i].score);
            text += " " + score_str;
            cv::Scalar color(0, 255, 0);  // 绿色
            cv::rectangle(orig_img_clone, box, color, 2);
            cv::Point textOrg(box.x, box.y - 10);
            cv::putText(orig_img_clone, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
        }
        std::string image_path_str = "face_det_result.png";
        const char* image_path = image_path_str.c_str();
        cv::imwrite(image_path, orig_img_clone);
        std::cout << "Draw result on" << image_path << " is finished." << std::endl;
    }
    
    
    // ret = release_retinaface_model(&rknn_app_ctx);
    // if (ret != 0) {
    //     printf("release_retinaface_model fail! ret=%d\n", ret);
    // }
    return result;
}

#ifdef __cplusplus
}
#endif