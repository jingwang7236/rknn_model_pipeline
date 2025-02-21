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


/* 定义类别字典 */
std::map<int, std::string> det_knife_category_map = {
    {0, "knife"}
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
 object_detect_result_list inference_det_knife_model(
    rknn_app_context_t* app_ctx,
    det_model_input input_data,
    model_inference_params params_,
    bool det_by_square,
    bool enable_logger)

 */
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif
object_detect_result_list inference_det_knife_model(rknn_app_context_t* app_ctx, det_model_input input_data, model_inference_params params_, bool det_by_square, bool enable_logger) {
    object_detect_result_list od_results;
    int ret = 0;

    if (det_by_square) {
        if (enable_logger) {
            printf("INFO: infer by square patch image\n");
        }
        ret = processDetBySquare(app_ctx, &od_results, input_data, params_.input_width, params_.input_height, params_.nms_threshold, params_.box_threshold, enable_logger);
    }
    else {
        if (enable_logger) {
            printf("INFO: infer by original image\n");
        }
        ImagePreProcess image_preprocess(input_data.width, input_data.height, params_.input_width, params_.input_height);
        auto convert_img = image_preprocess.Convert(convertDataToCvMat(input_data));
        ret = inference_yolov8_model(app_ctx, convert_img->ptr<unsigned char>(), &od_results, image_preprocess.get_letter_box(), det_knife_category_map.size(), params_.nms_threshold, params_.box_threshold, enable_logger);
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
            object_detect_result* det_result = &(od_results.results[i]);

            printf("%s @ (%d %d %d %d) %.3f\n", det_knife_category_map[det_result->cls_id].c_str(),
                det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom, det_result->prop);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            if (draw_box) {
                draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
                sprintf(text, "%s %.1f%%", det_knife_category_map[det_result->cls_id].c_str(), det_result->prop * 100);
                draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
            }
        }

        if (draw_box) {
            const char* image_path = "draw_result_knife.png";
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

// object_detect_result_list inference_det_knife_model(rknn_app_context_t *app_ctx, det_model_input input_data, const char* label_txt_path, bool enable_logger = true)
// {
//     object_detect_result_list od_results;
//     // const char* model_path = "model/yolov10s.rknn";
//     // const char *image_path = argv[2];

//     // cv::Mat orig_img_rgb;
//     // cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);
//     // cv::Mat orig_img 图片转为image_buffer_t格式
//     image_buffer_t src_image;
//     memset(&src_image, 0, sizeof(image_buffer_t));
//     src_image.width = input_data.width;
//     src_image.height = input_data.height;
//     src_image.format = IMAGE_FORMAT_RGB888;
//     src_image.size = input_data.width * input_data.height * input_data.channel;
//     src_image.virt_addr = (unsigned char*)malloc(src_image.size);
//     if (src_image.virt_addr == NULL) {
//         printf("malloc buffer size:%d fail!\n", src_image.size);
//         return od_results;
//     }
//     memcpy(src_image.virt_addr, input_data.data, src_image.size);

//     int ret;
//     // rknn_app_context_t rknn_app_ctx;
//     // memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

//     init_post_process(label_txt_path);

//     // ret = init_yolov10_model(model_path, &rknn_app_ctx);
//     // if (ret != 0)
//     // {
//     //     printf("init_yolov10_model fail! ret=%d model_path=%s\n", ret, model_path);
//     //     return od_results;
//     // }

//     // image_buffer_t src_image;
//     // memset(&src_image, 0, sizeof(image_buffer_t));
//     // ret = read_image(image_path, &src_image);

//     // if (ret != 0)
//     // {
//     //     printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
//     //     goto out;
//     // }

//     // object_detect_result_list od_results;

//     ret = inference_yolov8_model(app_ctx, &src_image, &od_results);
//     if (ret != 0)
//     {
//         printf("init_yolov8_model fail! ret=%d\n", ret);
//         return od_results;
//     }

//     // 画框
//     bool draw_box = true;
//     char text[256];
//     int count;
//     for (int i = 0; i < od_results.count; i++)
//     {
//         object_detect_result* det_result = &(od_results.results[i]);
//         std::string cls_name = coco_cls_to_name(det_result->cls_id, 1);
//        /* if (cls_name != "knife")
//         {
//             continue;
//         }*/
//         count++;
//         if (enable_logger) {
//             printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id, 1),
//                 det_result->box.left, det_result->box.top,
//                 det_result->box.right, det_result->box.bottom,
//                 det_result->prop);
//         }

//         int x1 = det_result->box.left;
//         int y1 = det_result->box.top;
//         int x2 = det_result->box.right;
//         int y2 = det_result->box.bottom;
//         if (draw_box)
//         {
//             draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
//             sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
//             draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
//         }
//     }
//     if (draw_box) {
//         const char* image_path = "result.png";
//         write_image(image_path, &src_image);
//         std::cout << "Draw result on" << image_path << " is finished." << std::endl;
//     }
//     od_results.count = count;

//     deinit_post_process();


//     if (src_image.virt_addr != NULL)
//     {
//         free(src_image.virt_addr);

//     }
//     return od_results;
// }
