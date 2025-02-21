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
#define INPUT_WIDTH_POSE_REN 640
#define INPUT_HEIGHT_POSE_REN 640
#define CLASS_NUM_POSE_REN 1
#define KPT_NUM_POSE_REN 17
#define RESULT_NUM_POSE_REN 8400
#define NMS_THRESH_POSE_REN 0.7
#define BOX_THRESH_POSE_REN 0.25

// 定义类别字典
std::map<int, std::string> pose_ren_category_map = {
    {0, "people"}
};

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
/*
 * 推理人体关键点检测模型。
 *
 * 功能：
 * - 执行人体关键点检测推理。
 *
 * 参数：
 * - app_ctx: RKNN 应用上下文，用于推理。
 * - input_data: 包含输入图像数据和尺寸的结构。
 * - enable_logger: 是否启用日志输出。
 *
 * 返回：
 * - object_detect_pose_result_list: 包含关键点检测结果的结构。
 */
#ifdef __cplusplus
extern "C" {          // 确保函数名称不会在导出时被修饰
#endif
object_detect_pose_result_list inference_pose_ren_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger) {
    object_detect_pose_result_list od_results;
    int ret = 0;

    ImagePreProcess pose_ren_image_preprocess(input_data.width, input_data.height, INPUT_WIDTH_POSE_REN, INPUT_HEIGHT_POSE_REN);
    auto convert_img = pose_ren_image_preprocess.Convert(convertDataToCvMat(input_data));

    ret = inference_yolov8_pose_mdoel(app_ctx, convert_img->ptr<unsigned char>(), &od_results, pose_ren_image_preprocess.get_letter_box(),
                                    CLASS_NUM_POSE_REN, KPT_NUM_POSE_REN, RESULT_NUM_POSE_REN,
                                    NMS_THRESH_POSE_REN, BOX_THRESH_POSE_REN, enable_logger);

    if (ret != 0) {
        printf("ERROR: det_kx model infer failed! ret=%d\n", ret);
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
        int skeleton[38] ={16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7}; 

        for (int i = 0; i < od_results.count; i++) {
            object_detect_pose_result* det_result = &(od_results.results[i]);

            // for (int j = 0; j < 17; j++){
            //     printf("%.2f, %.2f, %.lf\n", det_result->keypoints[j][0], det_result->keypoints[j][1], det_result->keypoints[j][2]);
            // }

            printf("%s @ (%d %d %d %d) %.3f\n", pose_ren_category_map[det_result->cls_id].c_str(),
                det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom, det_result->prop);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            if (draw_box) {
                draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
                sprintf(text, "%s %.1f%%", pose_ren_category_map[det_result->cls_id].c_str(), det_result->prop * 100);
                draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);

                for (int j = 0; j < 38/2; ++j){
                    draw_line(&src_image, (int)(det_result->keypoints[skeleton[2*j]-1][0]),(int)(det_result->keypoints[skeleton[2*j]-1][1]),
                    (int)(det_result->keypoints[skeleton[2*j+1]-1][0]),(int)(det_result->keypoints[skeleton[2*j+1]-1][1]),COLOR_ORANGE,3);
                }
                
                for (int j = 0; j < 17; ++j){
                    draw_circle(&src_image, (int)(det_result->keypoints[j][0]),(int)(det_result->keypoints[j][1]),1, COLOR_YELLOW,1);
                }
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
#ifdef __cplusplus
}
#endif
