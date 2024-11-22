// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov10.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "opencv2/opencv.hpp"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

bool draw_box = false;
int inference_person_det_model(cv::Mat orig_img, object_detect_result_list* od_results)
{
    const char* model_path = "model/yolov10s.rknn";
    // const char *image_path = argv[2];

    cv::Mat orig_img_rgb;
    // cv::Mat orig_img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);
    // cv::Mat orig_img 图片转为image_buffer_t格式
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = orig_img_rgb.cols;
    src_image.height = orig_img_rgb.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = orig_img_rgb.cols * orig_img_rgb.rows * orig_img_rgb.channels();
    src_image.virt_addr = (unsigned char *)malloc(src_image.size);
    if (src_image.virt_addr == NULL) {
      printf("malloc buffer size:%d fail!\n", src_image.size);
      return -1;
    }
    memcpy(src_image.virt_addr, orig_img_rgb.data, src_image.size);
    
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov10_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov10_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }
    
    // image_buffer_t src_image;
    // memset(&src_image, 0, sizeof(image_buffer_t));
    // ret = read_image(image_path, &src_image);
    
    // if (ret != 0)
    // {
    //     printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
    //     goto out;
    // }

    // object_detect_result_list od_results;

    ret = inference_yolov10_model(&rknn_app_ctx, &src_image, od_results);
    if (ret != 0)
    {
        printf("init_yolov10_model fail! ret=%d\n", ret);
        goto out;
    }

    // 画框和概率
    char text[256];
    int count;
    for (int i = 0; i < od_results->count; i++)
    {
        object_detect_result *det_result = &(od_results->results[i]);
        std::string cls_name = coco_cls_to_name(det_result->cls_id);
        if (cls_name != "person")
        {
            continue;
        }
        count++;
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
            det_result->box.left, det_result->box.top,
            det_result->box.right, det_result->box.bottom,
            det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        if (draw_box)
        {
            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        }
    }
    if (draw_box){
        write_image("out.png", &src_image);
    }
    od_results->count = count;
        

out:
    deinit_post_process();

    ret = release_yolov10_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov10_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);

    }
    return ret;
}
