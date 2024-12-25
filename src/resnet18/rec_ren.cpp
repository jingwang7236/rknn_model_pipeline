// yolov10 person detection

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
// #include "opencv2/opencv.hpp"
#include "resnet18.h"
#include "outer_model/model_func.hpp"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

#define RET_TOP_K 1

resnet_result inference_rec_person_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger=false)
{
    resnet_result od_results;
    // const char* model_path = "model/yolov10s.rknn";
    // const char *image_path = argv[2];

    // cv::Mat orig_img_rgb;
    // cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);
    // cv::Mat orig_img 图片转为image_buffer_t格式
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = input_data.width;
    src_image.height = input_data.height;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = input_data.width * input_data.height * input_data.channel;
    src_image.virt_addr = (unsigned char *)malloc(src_image.size);
    if (src_image.virt_addr == NULL) {
      printf("malloc buffer size:%d fail!\n", src_image.size);
      return od_results;
    }
    memcpy(src_image.virt_addr, input_data.data, src_image.size);
    
    int ret = inference_resnet_model(app_ctx, &src_image, &od_results, RET_TOP_K);
    if (ret != 0)
    {
        od_results.cls = -1;
        od_results.score = -1;
        printf("init_rec_ren_resnet_model fail! ret=%d\n", ret);
        return od_results;
    }

    return od_results;
}
