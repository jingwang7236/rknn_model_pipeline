// yolov10 person detection

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "yolov10.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
// #include "opencv2/opencv.hpp"
#include "outer_model/model_func.hpp"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

object_detect_result_list inference_person_det_model(rknn_app_context_t *app_ctx, det_model_input input_data, bool enable_logger=false)
{
    object_detect_result_list od_results;
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
    
    int ret;
    // rknn_app_context_t rknn_app_ctx;
    // memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    // ret = init_yolov10_model(model_path, &rknn_app_ctx);
    // if (ret != 0)
    // {
    //     printf("init_yolov10_model fail! ret=%d model_path=%s\n", ret, model_path);
    //     return od_results;
    // }
    
    // image_buffer_t src_image;
    // memset(&src_image, 0, sizeof(image_buffer_t));
    // ret = read_image(image_path, &src_image);
    
    // if (ret != 0)
    // {
    //     printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
    //     goto out;
    // }

    // object_detect_result_list od_results;

    ret = inference_yolov10_model(app_ctx, &src_image, &od_results);
    if (ret != 0)
    {
        printf("init_yolov10_model fail! ret=%d\n", ret);
        return od_results;
    }

    // 画框
    bool draw_box = false;
    char text[256];
    int count;
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        std::string cls_name = coco_cls_to_name(det_result->cls_id);
        if (cls_name != "person")
        {
            continue;
        }
        count++;
        if (enable_logger){
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
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
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        }
    }
    if (draw_box){
        const char* image_path = "person_det_result.png";
        write_image(image_path, &src_image);
        std::cout << "Draw result on" << image_path << " is finished." << std::endl;
    }
    od_results.count = count;

    deinit_post_process();

    // ret = release_yolov10_model(&rknn_app_ctx);
    // if (ret != 0)
    // {
    //     printf("release_yolov10_model fail! ret=%d\n", ret);
    // }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);

    }
    return od_results;
}
