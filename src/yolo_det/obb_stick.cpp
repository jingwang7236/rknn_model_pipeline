// yolov10 person detection

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "yolov8.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "opencv2/opencv.hpp"
#include "outer_model/model_func.hpp"
#include "yolo_image_preprocess.h"


std::map<int, std::string> obb_stick_category_map = {
    {0, "stick"}
};

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
/*
 * 推理枪支检测模型。
 *
 * 功能：
 * - 枪支检测推理。
 * - 支持裁剪为正方形块进行分块推理。
 *
 * 参数：
 * - app_ctx: RKNN 应用上下文，用于推理。
 * - input_data: 包含输入图像数据和尺寸的结构。
 * - model_inference_params: 模型推理参数，包含模型推理尺寸宽高，NMS阈值，检测框阈值。
 * - det_by_square: 是否以正方形分块方式进行检测。
 * - enable_logger: 是否启用日志输出。
 *
 * 返回：
 * - object_detect_result_list: 包含检测结果的结构。
 */

/*
object_detect_obb_result_list inference_obb_stick_model(
    rknn_app_context_t* app_ctx, 
    det_model_input input_data, 
    model_inference_params params_,
    bool det_by_square, 
    bool enable_logger) 

*/
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif
object_detect_obb_result_list inference_obb_stick_model(rknn_app_context_t* app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger) {
    object_detect_obb_result_list od_results;
    int ret = 0;

    if (det_by_square) {
        if (enable_logger) {
            printf("INFO: infer by square patch image\n");
        }
        ret = processDetBySquareObb(app_ctx, &od_results, input_data, params_.input_width, params_.input_height, params_.nms_threshold, params_.box_threshold, enable_logger);
    }
    else {
        if (enable_logger) {
            printf("INFO: infer by original image\n");
        }
        ImagePreProcess image_preprocess(input_data.width, input_data.height, params_.input_width, params_.input_height);
        auto convert_img = image_preprocess.Convert(convertDataToCvMat(input_data));
        
        ret = inference_yolov8_obb_model(app_ctx, convert_img->ptr<unsigned char>(), &od_results, image_preprocess.get_letter_box(), obb_stick_category_map.size(), params_.nms_threshold, params_.box_threshold, enable_logger);
    }

    if (ret != 0) {
        printf("ERROR: det_hand model infer failed! ret=%d\n", ret);
        return od_results;
    }

    // 画框
    if (enable_logger) {
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
            object_detect_obb_result* det_result = &(od_results.results[i]);

            printf("%s @ (%d %d %d %d angle=%f) %.3f\n", obb_stick_category_map[det_result->cls_id].c_str(),
                det_result->box.x, det_result->box.y,
                det_result->box.w, det_result->box.h,
                det_result->box.angle, det_result->prop);
            int x1 = det_result->box.x;
            int y1 = det_result->box.y;
            int w = det_result->box.w;
            int h = det_result->box.h;
            float angle = det_result->box.angle;

            if (draw_box) {
                draw_obb_rectangle(&src_image, x1, y1, w, h, angle, COLOR_BLUE, 3);
                sprintf(text, "%s %.1f%%", obb_stick_category_map[det_result->cls_id].c_str(), det_result->prop * 100);
                draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 8);
            }
        }

        if (draw_box) {
            const char* image_path = "draw_result_stick.png";
            write_image(image_path, &src_image);
            std::cout << "Draw result on " << image_path << " is finished." << std::endl;
        }

        // 释放内存
        free(src_image.virt_addr);
    }

    return od_results;
}
#ifdef __cplusplus
}
#endif
