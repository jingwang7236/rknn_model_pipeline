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
#include "opencv2/opencv.hpp"
#include "outer_model/model_func.hpp"

/*-------------------------------------------
                Defines
-------------------------------------------*/
// 定义类别字典
std::map<int, std::string> category_map = {
    {0, "hand"}
};

/*-------------------------------------------
          Helper Functions
-------------------------------------------*/

// 使用 OpenCV 填充图像至16的倍数
cv::Mat pad_image_to_multiple_of_16(const cv::Mat& img, int& pad_width, int& pad_height) {
    // 计算填充后的宽高
    int padded_width = (img.cols + 15) / 16 * 16;
    int padded_height = (img.rows + 15) / 16 * 16;

    // 计算需要填充的宽度和高度
    pad_width = padded_width - img.cols;
    pad_height = padded_height - img.rows;

    // 使用 OpenCV 的 copyMakeBorder 函数进行填充
    cv::Mat padded_img;
    cv::copyMakeBorder(img, padded_img, 0, pad_height, 0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    std::cout << "Image padded: original " << img.cols << "x" << img.rows
              << " -> padded " << padded_width << "x" << padded_height << std::endl;

    return padded_img;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

object_detect_result_list inference_det_hand_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger = true) {
    object_detect_result_list od_results;

    // 将输入数据转为 OpenCV 的 Mat 格式
    cv::Mat input_img(input_data.height, input_data.width, CV_8UC3, input_data.data);

    // 填充图像
    int pad_width = 0, pad_height = 0;
    cv::Mat padded_img = pad_image_to_multiple_of_16(input_img, pad_width, pad_height);

    // 准备填充后的图像数据
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = padded_img.cols;
    src_image.height = padded_img.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.size = padded_img.cols * padded_img.rows * 3;

    // 分配内存
    src_image.virt_addr = (unsigned char*)malloc(src_image.size);
    if (src_image.virt_addr == nullptr) {
        std::cerr << "Failed to allocate memory for src_image!" << std::endl;
        return od_results;
    }

    // 拷贝填充后的图像数据到分配的内存中
    memcpy(src_image.virt_addr, padded_img.data, src_image.size);

    // 推理
    int ret = inference_yolov8_model(app_ctx, &src_image, &od_results);
    if (ret != 0) {
        std::cerr << "inference_yolov8_model failed! ret=" << ret << std::endl;
        free(src_image.virt_addr);
        return od_results;
    }

    // 如果有填充，裁剪推理结果
    if (pad_width > 0 || pad_height > 0) {
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det_result = &(od_results.results[i]);
            det_result->box.left = std::max(0, det_result->box.left);
            det_result->box.top = std::max(0, det_result->box.top);
            det_result->box.right = std::min(input_data.width - 1, det_result->box.right);
            det_result->box.bottom = std::min(input_data.height - 1, det_result->box.bottom);
        }
        printf("Removed padding from results.\n");
    }

    // 画框
    bool draw_box = true;
    char text[256];
    int count = 0;
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result* det_result = &(od_results.results[i]);
        count++;

        if (enable_logger) {
            printf("%s @ (%d %d %d %d) %.3f\n", category_map[det_result->cls_id].c_str(),
                   det_result->box.left, det_result->box.top,
                   det_result->box.right, det_result->box.bottom,
                   det_result->prop);
        }

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        if (draw_box) {
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

    // 释放内存
    free(src_image.virt_addr);

    return od_results;
}
