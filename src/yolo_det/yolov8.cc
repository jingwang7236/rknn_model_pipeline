// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "yolov8.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

#include <sys/time.h>
#include "opencv2/opencv.hpp"

#include "yolo_postprocess.h"

static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
        "zp=%d, scale=%f\n",
        attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
        attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


static int convert_image_with_letterbox_opencv(rknn_app_context_t* app_ctx, const cv::Mat& orig_img, cv::Mat& dist_img, letterbox_t* letter_box)
{
    // 获取原始图像的宽度和高度
    int originalWidth = orig_img.cols;
    int originalHeight = orig_img.rows;
    // 获取目标图像的宽度和高度
    int targetWidth = app_ctx->model_width;
    int targetHeight = app_ctx->model_height;
    // 计算缩放比例
    float widthScale = (float)targetWidth / originalWidth;
    float heightScale = (float)targetHeight / originalHeight;
    float scale = (widthScale < heightScale) ? widthScale : heightScale;

    // 计算新的宽度和高度
    int newWidth = (int)(originalWidth * scale);
    int newHeight = (int)(originalHeight * scale);

    cv::Mat resize_img;
    cv::resize(orig_img, resize_img, cv::Size(newWidth, newHeight));
    // 计算需要填充的像素
    int left_pad = (targetWidth - newWidth) / 2;  // 居中贴图
    int right_pad = targetWidth - newWidth - left_pad;
    int top_pad = (targetHeight - newHeight) / 2;
    int bottom_pad = targetHeight - newHeight - top_pad;

    // 填充图像
    cv::copyMakeBorder(resize_img, dist_img, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::imwrite("input_scale.png", dist_img);

    letter_box->scale = scale;
    letter_box->x_pad = left_pad;
    letter_box->y_pad = top_pad;

    return 0;
}


int inference_yolov8_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results)
{
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];

    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值

    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // Post Process
    post_process_det(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    // dump_tensor_attr(od_results);

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}


int inference_yolov8_model_opencv(rknn_app_context_t* app_ctx, cv::Mat src_img, object_detect_result_list* od_results)
{
    int ret;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    int bg_color = 114;

    if ((!app_ctx) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    cv::Mat dist_img;
    ret = convert_image_with_letterbox_opencv(app_ctx, src_img, dist_img, &letter_box);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox_opencv fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = dist_img.cols * dist_img.rows * dist_img.channels() * sizeof(uint8_t);
    inputs[0].buf = dist_img.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }


    // Post Process
    post_process_det(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:
    return ret;
}


int inference_yolov8_obb_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_obb_result_list* od_results)
{
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // Default NMS threshold
    const float box_conf_threshold = BOX_THRESH; // Default box threshold
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        goto out;
    }

    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        goto out;
    }
    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run
    printf("rknn_run\n");
    int start_us, end_us;
    start_us = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    end_us = getCurrentTimeUs() - start_us;
    printf("rknn_run time=%.2fms, FPS = %.2f\n", end_us / 1000.f,
        1000.f * 1000.f / end_us);

    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }
    // Post Process
    start_us = getCurrentTimeUs();
    post_process_obb(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    end_us = getCurrentTimeUs() - start_us;
    printf("post_process time=%.2fms, FPS = %.2f\n", end_us / 1000.f,
        1000.f * 1000.f / end_us);
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}


int inference_yolov8_obb_model_opencv(rknn_app_context_t* app_ctx, cv::Mat src_img, object_detect_obb_result_list* od_results)
{
    int ret;
    cv::Mat dist_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // Default NMS threshold
    const float box_conf_threshold = BOX_THRESH; // Default box threshold
    int bg_color = 114;

    if ((!app_ctx) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    ret = convert_image_with_letterbox_opencv(app_ctx, src_img, dist_img, &letter_box);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox_opencv fail! ret=%d\n", ret);
        return -1;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = dist_img.cols * dist_img.rows * dist_img.channels();
    inputs[0].buf = dist_img.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("rknn_run\n");
    int start_us, end_us;
    start_us = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    end_us = getCurrentTimeUs() - start_us;
    printf("rknn_run time=%.2fms, FPS = %.2f\n", end_us / 1000.f, 1000.f * 1000.f / end_us);

    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    // Post Process
    start_us = getCurrentTimeUs();
    post_process_obb(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    end_us = getCurrentTimeUs() - start_us;
    printf("post_process time=%.2fms, FPS = %.2f\n", end_us / 1000.f, 1000.f * 1000.f / end_us);

    // Release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    return ret;
}



int inference_yolov8_pose_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_pose_result_list* od_results)
{
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // Default NMS threshold
    const float box_conf_threshold = BOX_THRESH; // Default box threshold
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        goto out;
    }

    // letterbox
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        goto out;
    }
    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run
    printf("rknn_run\n");
    int start_us, end_us;
    start_us = getCurrentTimeUs();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    end_us = getCurrentTimeUs() - start_us;
    printf("rknn_run time=%.2fms, FPS = %.2f\n", end_us / 1000.f,
        1000.f * 1000.f / end_us);

    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }
    // Post Process
    start_us = getCurrentTimeUs();
    post_process_pose(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
    end_us = getCurrentTimeUs() - start_us;
    printf("post_process time=%.2fms, FPS = %.2f\n", end_us / 1000.f,
        1000.f * 1000.f / end_us);
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}


int inference_yolov8_pose_model_opencv(rknn_app_context_t* app_ctx, cv::Mat src_img, object_detect_pose_result_list* od_results)
{
    int ret;
    cv::Mat dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // Default NMS threshold
    const float box_conf_threshold = BOX_THRESH; // Default box threshold
    int bg_color = 114;

    if ((!app_ctx) || (!od_results))
    {

int inference_yolov8_model(rknn_app_context_t* app_ctx, void* image_buf, object_detect_result_list* od_results, letterbox_t letter_box, float nms_threshold, float box_conf_threshold, bool enable_logger){
    int ret;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    

    if ((!app_ctx) || !(image_buf) || (!od_results))
    {
        printf("ERROR: Input app_ctx/image_buffer/od_results is null!\n");

        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));

    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    ret = convert_image_with_letterbox_opencv(app_ctx, src_img, dst_img, &letter_box);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox_opencv fail! ret=%d\n", ret);
        return -1;
    }


    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));
    
    // 妯″瀷寮�濮嬫帹鐞嗘椂闂存埑
    auto total_start_time = std::chrono::high_resolution_clock::now();

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = image_buf;


    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("ERROR: rknn_input_set fail! ret=%d\n", ret);

        return -1;
    }

    // Run

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("ERROR: rknn_run fail! ret=%d\n", ret);

        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);

    // 鎺ㄧ悊缁撴潫鏃堕棿
    auto inference_end_time = std::chrono::high_resolution_clock::now();

    if (ret < 0)
    {
        printf("ERROR: rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // Post Process
    post_process_det_hw(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results, 1);

    // dump_tensor_attr(od_results);

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    // 注释
    if (enable_logger){
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration<double, std::milli>(inference_end_time - total_start_time);
        auto postprocess_duration = std::chrono::duration<double, std::milli>(total_end_time - inference_end_time);
        auto total_duration = std::chrono::duration<double, std::milli>(total_end_time - total_start_time);

        printf("INFO: total infer time %.2f ms: model time is %.2f ms and postprocess time is %.2fms\n",
            total_duration.count(), inference_duration.count(), postprocess_duration.count());
    }

    out:
    
    return ret;
}

