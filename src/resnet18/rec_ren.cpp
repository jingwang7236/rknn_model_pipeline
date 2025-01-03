// resnet18 person recognition

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
#include "opencv2/opencv.hpp"
#include "resnet18.h"
#include "outer_model/model_func.hpp"
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb_image_resize2.h"

/*-------------------------------------------
                Constants
-------------------------------------------*/
#define RET_TOP_K 1
#define IMG_HEIGHT 320
#define IMG_WIDTH 160
#define READ_IMAGE_TYPE STBIR_RGB

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

resnet_result inference_rec_person_resnet18_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger=false)
{
    resnet_result od_results;
    memset(&od_results, 0, sizeof(resnet_result));

    // 分配内存用于存储调整大小后的图像
    unsigned char* resized_data = (unsigned char*)malloc(IMG_WIDTH * IMG_HEIGHT * input_data.channel);

    if (!resized_data) {
        printf("Failed to allocate memory for resized image\n");
        od_results.cls = -3;
        od_results.score = -3;
        return od_results;
    }

    // 调整图像大小
    if (!stbir_resize_uint8_linear(input_data.data, input_data.width, input_data.height, 0,
                                   resized_data, IMG_WIDTH, IMG_HEIGHT, 0, READ_IMAGE_TYPE)) {
        printf("Failed to resize image\n");
        free(resized_data);
        od_results.cls = -2;
        od_results.score = -2;
        return od_results;
    }


    /*
    // Load image data into OpenCV Mat
    cv::Mat input_img(input_data.height, input_data.width, CV_8UC3, input_data.data);  // Assuming RGB format

    // Resize image using OpenCV
    cv::Mat resized_img;
    cv::resize(input_img, resized_img, cv::Size(IMG_WIDTH, IMG_HEIGHT), 0, 0, cv::INTER_LINEAR); // Using linear interpolation for resizing
    */

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    src_image.width = IMG_WIDTH;
    src_image.height = IMG_HEIGHT;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = IMG_WIDTH * IMG_HEIGHT * input_data.channel;
    // src_image.virt_addr = resized_img.data;
    src_image.virt_addr = resized_data;
    
    int ret = inference_resnet_model(app_ctx, &src_image, &od_results, RET_TOP_K);
    if (ret != 0)
    {
        od_results.cls = -1;
        od_results.score = -1;
        printf("init_rec_ren_resnet_model fail! ret=%d\n", ret);
        free(resized_data);
        return od_results;
    }

    // Clean up resources
    free(resized_data);

    return od_results;
}
