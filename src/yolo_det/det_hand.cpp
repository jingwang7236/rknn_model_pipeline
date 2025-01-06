// yolov10 person detection

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <map>
#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
// #include "opencv2/opencv.hpp"
#include "outer_model/model_func.hpp"

/*-------------------------------------------
                Defines
-------------------------------------------*/
// 定义类别字典
std::map<int, std::string> category_map = {
    {0, "hand"}
};

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

object_detect_result_list inference_det_hand_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = true)
{
    object_detect_result_list od_results;
    image_buffer_t src_image;

    // 初始化结构体为 0
    memset(&src_image, 0, sizeof(image_buffer_t));

    // 设置图像的宽度、高度、格式和大小
    src_image.width = input_data.width;
    src_image.height = input_data.height;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = input_data.width * input_data.height * input_data.channel;

    // 分配内存
    src_image.virt_addr = (unsigned char*)malloc(src_image.size);

    if (src_image.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", src_image.size);
        return od_results;
    }

    memcpy(src_image.virt_addr, input_data.data, src_image.size);
    int ret = inference_yolov8_model(app_ctx, &src_image, &od_results);

    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d\n", ret);
        return od_results;
    }

    // 画框
    bool draw_box = true;
    char text[256];
    int count;
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result* det_result = &(od_results.results[i]);
        count++;

        if (enable_logger) {
            std::cout << det_result->cls_id << std::endl;
            std::cout << category_map[0] << std::endl;
            printf("%s @ (%d %d %d %d) %.3f\n", category_map[det_result->cls_id].c_str(),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
        }

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        if (draw_box)
        {
            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
            sprintf(text, "%s %.1f%%", category_map[det_result->cls_id].c_str(), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        }
    }
    if (draw_box) {
        const char* image_path = "./data/draw_result.png";
        write_image(image_path, &src_image);
        std::cout << "Draw result on " << image_path << " is finished." << std::endl;
    }

    od_results.count = count;

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);

    }
    return od_results;
}
