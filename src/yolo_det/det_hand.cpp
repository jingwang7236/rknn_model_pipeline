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
#include "yolo_image_preprocess.h"

/*-------------------------------------------
                Defines
-------------------------------------------*/
#define INPUT_WIDTH_DET_HAND 800
#define INPUT_HEIGHT_DET_HAND 448
#define NMS_THRESH_DET_HAND 0.7
#define BOX_THRESH_DET_HAND 0.25

// 定义类别字典
std::map<int, std::string> det_hand_category_map = {
    {0, "hand"}
};

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
/*
 * 推理手部检测模型。
 *
 * 功能：
 * - 执行手部检测推理。
 * - 支持裁剪为正方形块进行分块推理。
 *
 * 参数：
 * - app_ctx: RKNN 应用上下文，用于推理。
 * - input_data: 包含输入图像数据和尺寸的结构。
 * - det_by_square: 是否以正方形分块方式进行检测。
 * - enable_logger: 是否启用日志输出。
 *
 * 返回：
 * - object_detect_result_list: 包含检测结果的结构。
 */

object_detect_result_list inference_det_hand_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool det_by_square, bool enable_logger) {
    object_detect_result_list od_results;
    int ret = 0;

    if (det_by_square && INPUT_WIDTH_DET_HAND == INPUT_HEIGHT_DET_HAND){
        if (enable_logger){
            printf("INFO: infer by square patch image\n");
        }
        ret = processDetBySquare(app_ctx, &od_results, input_data, INPUT_WIDTH_DET_HAND, INPUT_HEIGHT_DET_HAND, NMS_THRESH_DET_HAND, BOX_THRESH_DET_HAND, enable_logger);
    }else{
        if (enable_logger){
            printf("INFO: infer by original image\n");
        }
        ImagePreProcess det_hand_image_preprocess(input_data.width, input_data.height, INPUT_WIDTH_DET_HAND, INPUT_HEIGHT_DET_HAND);
        auto convert_img = det_hand_image_preprocess.Convert(convertDataToCvMat(input_data));
        // cv::Mat img_rgb = cv::Mat::zeros(INPUT_WIDTH_DET_HAND, INPUT_HEIGHT_DET_HAND, convert_img->type());
        // convert_img->copyTo(img_rgb);
        ret = inference_yolov8_model(app_ctx, convert_img->ptr<unsigned char>(), &od_results, det_hand_image_preprocess.get_letter_box(), NMS_THRESH_DET_HAND, BOX_THRESH_DET_HAND, enable_logger);
    }
    
    if (ret != 0) {
        printf("ERROR: det_hand model infer failed! ret=%d\n", ret);
        return od_results;
    }

    // 画框
    if (enable_logger){
        // 准备填充后的图像数据
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = input_data.width;
        src_image.height = input_data.height;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = input_data.width * input_data.height * 3;

        // 分配内存
        src_image.virt_addr = (unsigned char*)malloc(src_image.size);
        if (src_image.virt_addr == nullptr) {
            printf("ERROR: Failed to allocate memory for src_image!\n");
            return od_results;
        }

        // 拷贝填充后的图像数据到分配的内存中
        memcpy(src_image.virt_addr, input_data.data, src_image.size);

        bool draw_box = true;       
        char text[256];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result* det_result = &(od_results.results[i]);

            printf("%s @ (%d %d %d %d) %.3f\n", det_hand_category_map[det_result->cls_id].c_str(),
                det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom, det_result->prop);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            if (draw_box) {
                draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
                sprintf(text, "%s %.1f%%", det_hand_category_map[det_result->cls_id].c_str(), det_result->prop * 100);
                draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
            }
        }

        if (draw_box) {
            const char* image_path = "./data/draw_result.png";
            write_image(image_path, &src_image);
            std::cout << "Draw result on " << image_path << " is finished." << std::endl;
        }

        // 释放内存
        free(src_image.virt_addr);
    }

    return od_results;
}
